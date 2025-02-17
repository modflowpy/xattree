import numpy as np
from numpy.typing import NDArray
from xarray import DataTree

from xattree import array, child, dim, xattree


@xattree
class Grid:
    rows: int = dim(coord="j", scope="root", default=3)
    cols: int = dim(coord="i", scope="root", default=3)


@xattree
class Arrs:
    arr: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))


@xattree
class Root:
    grid = child(Grid)
    arrs = child(Arrs)


def test_access():
    """
    `attrs` attribute access should still work as expected.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(root)

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


def test_parent():
    """
    A `parent` reference should be added to instances
    allowing programmatic navigation through the tree
    without needing to access the `data` attribute.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(root)

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
    arrs = Arrs(root)

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
    arrs = Arrs(root)
    assert arrs.data["arr"].shape == (3, 3)


def test_array_expansion_explicit():
    """
    Arrays with scalar default values and declared dimensions
    should be expanded to the specified shape when dimensions
    are explicitly provided.
    """
    arrs = Arrs(dims={"rows": 3, "cols": 3})
    assert arrs.data["arr"].shape == (3, 3)
