import numpy as np
import pytest
from numpy.typing import NDArray
from xarray import DataTree

from xattree import ROOT, _get_xatspec, array, dim, field, xattree


@xattree
class Grid:
    rows: int = dim(name="row", scope=ROOT, default=3)
    cols: int = dim(name="col", scope=ROOT, default=3)
    nodes: int = dim(name="node", scope=ROOT, init=False)

    def __attrs_post_init__(self):
        self.nodes = self.rows * self.cols


@xattree
class Arrs:
    arr: NDArray[np.float64] = array(default=0.0, dims=("row", "col"))


@xattree
class Root:
    grid: Grid = field()
    arrs: Arrs = field()


def test_meta():
    xatspec = _get_xatspec(Root)
    assert set(xatspec.coords.keys()) == {"row", "col", "node"}


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
    assert root.data.dims["row"] == 3
    assert root.data.dims["col"] == 3
    assert root.data.dims["node"] == 9
    assert grid.data.dims["row"] == 3
    assert grid.data.dims["col"] == 3
    assert grid.data.dims["node"] == 9
    assert arrs.data.dims["row"] == 3
    assert arrs.data.dims["col"] == 3
    assert arrs.data.dims["node"] == 9
    assert np.array_equal(root.data.coords["row"], np.arange(3))
    assert np.array_equal(root.data.coords["col"], np.arange(3))
    assert np.array_equal(root.data.coords["node"], np.arange(9))
    assert np.array_equal(grid.data.coords["row"], np.arange(3))
    assert np.array_equal(grid.data.coords["col"], np.arange(3))
    assert np.array_equal(grid.data.coords["node"], np.arange(9))
    assert np.array_equal(arrs.data.coords["row"], np.arange(3))
    assert np.array_equal(arrs.data.coords["col"], np.arange(3))
    assert np.array_equal(arrs.data.coords["node"], np.arange(9))


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
    with pytest.raises(TypeError, match=r".*cannot be assigned to a DataTree.*"):
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
