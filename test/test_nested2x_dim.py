"""The example shown in the readme."""

import numpy as np
import pytest
from numpy.typing import NDArray
from xarray import DataTree

from xattree import ROOT, array, dim, field, xattree


@xattree
class Grid:
    rows: int = dim(name="row", scope=ROOT, default=3)
    cols: int = dim(name="col", scope=ROOT, default=3)
    # TODO test a dim scoped to mid


@xattree
class Arrs:
    arr: NDArray[np.float64] = array(default=0.0, dims=("row", "col"))


@xattree
class Mid:
    grid: Grid = field()
    arrs: Arrs = field()


@xattree
class Root:
    mid: Mid = field()


def test_access():
    """
    `attrs` attribute access should still work as expected.
    """
    grid = Grid()
    mid = Mid(grid=grid)
    arrs = Arrs(parent=mid)
    root = Root(mid=mid)

    assert root.mid.grid is grid
    assert root.mid.arrs is arrs
    assert root.mid.grid == grid
    assert root.mid.arrs == arrs
    assert grid.rows == 3
    assert grid.cols == 3
    assert isinstance(root.data, DataTree)
    assert isinstance(grid.data, DataTree)
    assert isinstance(arrs.data, DataTree)
    assert root.mid.grid.data is grid.data
    assert root.mid.arrs.data is arrs.data
    assert root.data.dims["row"] == 3
    assert root.data.dims["col"] == 3
    assert grid.data.dims["row"] == 3
    assert grid.data.dims["col"] == 3
    assert arrs.data.dims["row"] == 3
    assert arrs.data.dims["col"] == 3
    assert np.array_equal(grid.data.coords["row"], np.arange(3))
    assert np.array_equal(grid.data.coords["col"], np.arange(3))
    assert np.array_equal(arrs.data.coords["row"], np.arange(3))
    assert np.array_equal(arrs.data.coords["col"], np.arange(3))


def test_mutate_array():
    """
    `attrs` array attributes should be mutable, with all
    mutations reflected in the data tree and vice versa.
    Modifications to arrays must go through `values` as
    expected for `xarray.DataArray`.
    """
    grid = Grid()
    mid = Mid(grid=grid)
    arrs = Arrs(parent=mid)
    root = Root(mid=mid)

    arr = np.ones(arrs.arr.shape)
    with pytest.raises(TypeError, match=r".*cannot be assigned to a DataTree.*"):
        arrs.arr = arr
    arrs.arr.values = np.ones(arr.shape)
    assert np.array_equal(arrs.arr, arr)
    assert np.array_equal(arrs.data.arr, arr)
    arrs.data.arr.values = np.ones(arr.shape) * 2
    assert np.array_equal(arrs.arr, arr * 2)
    assert np.array_equal(arrs.data.arr, arr * 2)
    assert np.array_equal(root.mid.arrs.arr, arr * 2)


def test_mutate_child():
    """
    `attrs` child attributes should be mutable, with all
    mutations reflected in the data tree and vice versa.
    """

    grid = Grid()
    mid = Mid(grid=grid)
    root = Root(mid=mid)
    grid2 = Grid()
    mid.grid = grid2

    assert mid.grid is grid2
    assert mid.data.grid is grid2.data
    assert root.mid.grid is grid2
    assert root.mid.data.grid is grid2.data


def test_parent():
    """
    A `parent` reference should be added to instances
    allowing programmatic navigation through the tree
    without needing to access the `data` attribute.
    """
    grid = Grid()
    mid = Mid(grid=grid)
    arrs = Arrs(parent=mid)
    root = Root(mid=mid)

    assert grid.parent is mid
    assert arrs.parent is mid
    assert root.mid.parent is root
    assert root.mid.arrs.parent is mid
    assert root.mid.grid.data.parent is mid.data


def test_array_expansion_inherit():
    """
    Arrays with scalar default values and declared dimensions
    should be expanded to the specified shape when dimensions
    are inherited from the root node.
    """
    grid = Grid()
    mid = Mid(grid=grid)
    root = Root(mid=mid)
    arrs = Arrs(parent=mid)

    assert arrs.data["arr"].shape == (3, 3)
    assert np.array_equal(root.mid.arrs.data["arr"], arrs.data["arr"])
