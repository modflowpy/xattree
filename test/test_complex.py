import numpy as np
from attrs import Factory, define, field
from numpy.typing import NDArray
from xarray import DataTree

from xattree import xattree


@xattree
@define(slots=False)
class Grid:
    rows: int = field(
        default=3,
        metadata={
            "dim": {
                "coord": "j",
                "scope": "root",
            }
        },
    )
    cols: int = field(
        default=3,
        metadata={
            "dim": {
                "coord": "i",
                "scope": "root",
            },
        },
    )


@xattree
@define(slots=False)
class Arrs:
    arr: NDArray[np.float64] = field(
        default=0.0, metadata={"dims": ("rows", "cols")}
    )


@xattree
@define(slots=False)
class Root:
    grid: Grid = field(default=Factory(Grid), metadata={"bind": True})
    arrs: Arrs = field(default=Factory(Arrs), metadata={"bind": True})


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
    # TODO: custom callable for array eq comparison?
    # https://www.attrs.org/en/stable/comparison.html
    # assert root.arrs == arrs
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
    # TODO fix parent reference for nested nodes
    # which were constructed "bottom-up", i.e.,
    # passed as arguments to the parent node's
    # initializer instead of the parent node
    # passed as an argument to its initialiizer.
    # assert root.grid.data.parent is root.data
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
