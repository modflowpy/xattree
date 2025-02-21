"""The example shown in the readme."""

import numpy as np
import pytest
from attrs import field
from numpy.typing import NDArray
from xarray import DataTree

from xattree import array, dim, xattree


@xattree
class Grid:
    rows: int = dim(coord="j", scope="root", default=3)
    cols: int = dim(coord="i", scope="root", default=3)


@xattree
class Arrs:
    arr: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))


@xattree
class Root:
    grid: Grid = field()
    arrs: Arrs = field()


def test_access():
    """
    `attrs` attribute access should still work as expected.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(parent=root)

    assert root.grid is grid
    assert root.arrs is arrs
    assert root.grid == grid
    assert root.arrs == arrs
    assert grid.rows == 3
    assert grid.cols == 3
    assert isinstance(root.data, DataTree)
    assert isinstance(grid.data, DataTree)
    assert isinstance(arrs.data, DataTree)
    assert root.grid.data is grid.data
    assert root.arrs.data is arrs.data


def test_mutate_array():
    """
    `attrs` array attributes should be mutable, with all
    mutations reflected in the data tree and vice versa.
    Modifications to arrays must go through `values` as
    expected for `xarray.DataArray`.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(parent=root)

    arr = np.ones(arrs.arr.shape)
    with pytest.raises(
        TypeError, match=r".*cannot be assigned to a DataTree.*"
    ):
        arrs.arr = arr
    arrs.arr.values = np.ones(arr.shape)
    assert np.array_equal(arrs.arr, arr)
    assert np.array_equal(arrs.data.arr, arr)
    arrs.data.arr.values = np.ones(arr.shape) * 2
    assert np.array_equal(arrs.arr, arr * 2)
    assert np.array_equal(arrs.data.arr, arr * 2)


def test_mutate_child():
    """
    `attrs` child attributes should be mutable, with all
    mutations reflected in the data tree and vice versa.
    """

    grid = Grid()
    root = Root(grid=grid)
    grid2 = Grid()
    root.grid = grid2

    assert root.grid is grid2
    assert root.data.grid is grid2.data


def test_parent():
    """
    A `parent` reference should be added to instances
    allowing programmatic navigation through the tree
    without needing to access the `data` attribute.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(parent=root)

    assert grid.parent is root
    assert arrs.parent is root
    assert root.grid.parent is root
    assert root.arrs.parent is root
    assert root.grid.data.parent is root.data
    assert root.arrs.data.parent is root.data


def test_dims_and_coords():
    """
    Root-scoped dimensions/coordinates should be added to the
    root node's dataset and inherited by child nodes.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(parent=root)

    assert root.data.dims["rows"] == 3
    assert root.data.dims["cols"] == 3
    assert grid.data.dims["rows"] == 3
    assert grid.data.dims["cols"] == 3
    assert arrs.data.dims["rows"] == 3
    assert arrs.data.dims["cols"] == 3
    assert np.array_equal(grid.data.coords["i"], np.arange(3))
    assert np.array_equal(grid.data.coords["j"], np.arange(3))
    assert np.array_equal(arrs.data.coords["i"], np.arange(3))
    assert np.array_equal(arrs.data.coords["j"], np.arange(3))


def test_array_expansion_inherit():
    """
    Arrays with scalar default values and declared dimensions
    should be expanded to the specified shape when dimensions
    are inherited from the root node.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(parent=root)
    assert arrs.data["arr"].shape == (3, 3)
