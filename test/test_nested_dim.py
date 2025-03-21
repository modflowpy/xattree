import numpy as np
import pytest
from numpy.typing import NDArray
from xarray import DataTree

from xattree import ROOT, _get_xatspec, array, dim, field, xattree


@xattree
class Grid:
    rows: int = dim(scope=ROOT, default=3)
    cols: int = dim(scope=ROOT, default=3)
    nodes: int = dim(scope=ROOT, init=False)

    def __attrs_post_init__(self):
        self.nodes = self.rows * self.cols


@xattree
class Arrs:
    arr: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))


@xattree
class Root:
    grid: Grid = field()
    arrs: Arrs = field()


def test_meta():
    xatspec = _get_xatspec(Root)
    assert set(xatspec.coords.keys()) == {"rows", "cols", "nodes"}


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
    assert root.data.dims["rows"] == 3
    assert root.data.dims["cols"] == 3
    assert root.data.dims["nodes"] == 9
    assert grid.data.dims["rows"] == 3
    assert grid.data.dims["cols"] == 3
    assert grid.data.dims["nodes"] == 9
    assert arrs.data.dims["rows"] == 3
    assert arrs.data.dims["cols"] == 3
    assert arrs.data.dims["nodes"] == 9
    assert np.array_equal(root.data.coords["rows"], np.arange(3))
    assert np.array_equal(root.data.coords["cols"], np.arange(3))
    assert np.array_equal(root.data.coords["nodes"], np.arange(9))
    assert np.array_equal(grid.data.coords["rows"], np.arange(3))
    assert np.array_equal(grid.data.coords["cols"], np.arange(3))
    assert np.array_equal(grid.data.coords["nodes"], np.arange(9))
    assert np.array_equal(arrs.data.coords["rows"], np.arange(3))
    assert np.array_equal(arrs.data.coords["cols"], np.arange(3))
    assert np.array_equal(arrs.data.coords["nodes"], np.arange(9))


def test_replace_array():
    """
    `attrs` array attributes should be mutable, with all
    mutations reflected in the data tree and vice versa.
    Modifications directly to the `DataArray` must still
    go through `values` as `xarray` requires.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(parent=root)

    arr = np.ones(arrs.arr.shape)
    arrs.arr = arr
    arrs.arr.values = np.ones(arr.shape)
    assert np.array_equal(arrs.arr, arr)
    assert np.array_equal(arrs.data.arr, arr)
    arrs.data.arr.values = np.ones(arr.shape) * 2
    assert np.array_equal(arrs.arr, arr * 2)
    assert np.array_equal(arrs.data.arr, arr * 2)


def test_replace_child():
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


def test_top_down_misaligned_raises():
    """
    When components are constructed top-down (i.e. parents first)
    and a child component's dimensions disagree with the parent's
    inherited dimensions, expect an xarray alignment error raised.
    """
    root = Root()
    with pytest.raises(ValueError):
        Grid(parent=root, rows=4, cols=4)
