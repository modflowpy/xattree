import numpy as np
import pytest
from numpy.typing import NDArray
from xarray import DataTree

from xattree import ROOT, array, asdict, dim, field, get_xatspec, xattree


@xattree
class Grid:
    rows: int = dim(scope=ROOT, default=3)
    cols: int = dim(scope=ROOT, default=3)
    nodes: int = dim(scope=ROOT, init=False)

    def __attrs_post_init__(self):
        self.nodes = self.rows * self.cols


@xattree
class Arrs:
    a: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))


@xattree
class Root:
    grid: Grid = field()
    arrs: Arrs = field()


def test_meta():
    spec = get_xatspec(Root)
    keys = set(spec.dims.keys())
    assert "rows" in keys
    assert "cols" in keys
    assert "nodes" in keys


def test_access():
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

    a = np.ones(arrs.a.shape)
    arrs.a = a
    arrs.a.values = np.ones(a.shape)
    assert np.array_equal(arrs.a, a)
    assert np.array_equal(arrs.data.a, a)
    arrs.data.a.values = np.ones(a.shape) * 2
    assert np.array_equal(arrs.a, a * 2)
    assert np.array_equal(arrs.data.a, a * 2)


def test_replace_orphan_child():
    """
    `attrs` child attributes should be mutable, with all
    mutations reflected in the data tree and vice versa.
    A child node can be replaced with a new instance of
    the same type as long as it does not have a parent.
    """

    grid = Grid()
    root = Root(grid=grid)
    grid2 = Grid()
    root.grid = grid2

    assert root.grid is grid2
    assert root.data.grid is grid2.data


def test_replace_non_orphan_child_raises():
    """
    A child child node cannot be replaced with a node
    that already has a parent."""

    grid = Grid()
    root = Root(grid=grid)
    root2 = Root(grid=grid)
    arrs = Arrs(parent=root)
    with pytest.raises(AttributeError, match=r"already has a parent"):
        root2.arrs = arrs


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
    assert arrs.data["a"].shape == (3, 3)


def test_top_down_misaligned_raises():
    """
    When components are constructed top-down (i.e. parents first)
    and a child component's dimensions disagree with the parent's
    inherited dimensions, expect an xarray alignment error raised.
    """
    root = Root()
    with pytest.raises(ValueError):
        Grid(parent=root, rows=4, cols=4)


def test_asdict():
    """
    The `asdict` function should return a dictionary representation
    of the xattree instance, including nested structures.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(parent=root)

    result = asdict(root)
    assert isinstance(result, dict)
    assert "grid" in result
    assert "arrs" in result
    assert result["grid"]["rows"] == 3
    assert result["grid"]["cols"] == 3
    assert result["arrs"]["a"].shape == (3, 3)
    assert np.array_equal(asdict(arrs)["a"], result["arrs"]["a"])
