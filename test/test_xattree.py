import numpy as np
import pytest
from attrs import Factory, define, field
from numpy.typing import NDArray
from xarray import DataTree

from xattree import DimsNotFound, ExpandFailed, xattree


@xattree
@define(slots=False)
class Foo:
    row: int = field(metadata={"dim": {"coord": "j"}})
    col: int = field(metadata={"dim": {"coord": "i"}})


@xattree
@define(slots=False)
class Bar:
    i: NDArray[np.int64] = field(metadata={"coord": {"dim": "col"}})
    j: NDArray[np.int64] = field(metadata={"coord": {"dim": "row"}})


def test_attrs_basics():
    """`attrs` class attributes/behaviors should still work as expected."""
    foo = Foo(row=3, col=3)
    assert foo.row == 3
    assert foo.col == 3

    bar = Bar(i=np.arange(3), j=np.arange(3))
    assert np.array_equal(bar.i, np.arange(3))
    assert np.array_equal(bar.j, np.arange(3))


def test_tree_basics():
    """
    A `DataTree` should be attached to `.data` on the nested class instances.
    Scalars should be stored as attributes, arrays as variables.
    """
    foo = Foo(row=3, col=3)
    bar = Bar(i=np.arange(3), j=np.arange(3))

    assert isinstance(foo.data, DataTree)
    assert isinstance(bar.data, DataTree)
    assert foo.data.attrs["row"] == 3
    assert foo.data.attrs["col"] == 3
    assert np.array_equal(bar.data.i, np.arange(3))
    assert np.array_equal(bar.data.j, np.arange(3))


def test_tree_dims_and_coords():
    """
    The `DataTree` should have dimensions and coordinates as configured in
    the `dim` and `coord` metadata.
    """
    foo = Foo(row=3, col=3)
    bar = Bar(i=np.arange(3), j=np.arange(3))

    assert foo.data.dims["row"] == 3
    assert foo.data.dims["col"] == 3
    assert bar.data.dims["row"] == 3
    assert bar.data.dims["col"] == 3


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


def test_nested_attrs_basics():
    """
    Nested `attrs` class attributes/behaviors should still work as expected.
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


def test_nested_attrs_traversal():
    """
    A `parent` reference should be added to nested instances, allowing for
    programmatic navigation of the tree.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(root)

    assert grid.parent is root
    assert root.grid is grid
    assert root.grid == grid
    assert arrs.parent is root
    assert root.grid.parent is root
    assert root.arrs.parent is root


def test_nested_tree_basics():
    """
    A `DataTree` should be attached to `.data` on the nested class instances,
    allowing for programmatic navigation of the tree.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(root)

    assert isinstance(root.data, DataTree)
    assert isinstance(grid.data, DataTree)
    assert isinstance(arrs.data, DataTree)

    assert root.grid.data is grid.data
    assert root.arrs.data is arrs.data
    assert root.grid.data.parent is not root.data
    assert root.arrs.data.parent is root.data


def test_nested_tree_dims_and_coords():
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


def test_array_expansion_inherited():
    """
    The `Arrs.arr` attribute should be expanded to a 3x3 array.
    This should work when the tree is built bottom-up.
    """
    grid = Grid()
    root = Root(grid=grid)
    arrs = Arrs(root)
    assert arrs.data["arr"].shape == (3, 3)


def test_array_expansion_explicit():
    """
    The `Arrs.arr` attribute should be expanded to a 3x3 array.
    This should work when dimensions are explicitly provided.
    """
    arrs = Arrs(dims={"rows": 3, "cols": 3})
    assert arrs.data["arr"].shape == (3, 3)


def test_dims_not_found_strict():
    """
    When dims can't be found for an array variable and `strict=True`,
    an error should be raised.
    """
    with pytest.raises(
        DimsNotFound, match=r".*failed dim resolution: rows, cols.*"
    ):
        Arrs(strict=True)


def test_dims_not_found_relaxed():
    """
    When an array's value is not initialized, and dims can't be found, and
    `strict=False`, raise no error, but don't set the array on the dataset.
    """
    arrs = Arrs(strict=False)
    assert not any(arrs.data)


@xattree
@define(slots=False)
class BadArrs:
    arr: NDArray[np.float64] = field(default=0.0)


@xattree
@define(slots=False)
class BadRoot:
    arrs: BadArrs = field(default=Factory(BadArrs), metadata={"bind": True})


def test_no_dims_expand_fails():
    """
    When no dims are specified for an array variable, a default
    value may not be provided (since expansion is impossible).
    """
    with pytest.raises(ExpandFailed, match=r".*can't expand; no dims.*"):
        BadRoot()
