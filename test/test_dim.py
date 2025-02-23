
import numpy as np
import pytest

from xattree import dim, xattree


def test_dim_coord():
    """
    By default, an unmodified dimension field becomes a dimension
    coordinate: an eponymous coordinate array is created with the
    same name as the field, and accessing an attribute returns the
    coordinate array. To access the dimension size use `data.dims`.
    """

    @xattree
    class Foo:
        t: int = dim()

    n = 3
    arr = np.arange(n)
    foo = Foo(t=n)
    assert foo.data.dims["t"] == n
    assert np.array_equal(foo.t, arr)
    assert np.array_equal(foo.data.t, arr)


def test_dim_coord_with_coord_alias():
    """
    Test a dimension coordinate with a coordinate alias. This
    is like a dimension coordinate, with the coordinate array
    renamed to `coord`.

    TODO: file an issue on xarray about associating a coord
    and a dim with different names?
    """

    @xattree
    class Foo:
        rows: int = dim(coord="j")
        cols: int = dim(coord="i")

    n = 3
    arr = np.arange(n)
    foo = Foo(rows=n, cols=n)
    assert foo.rows == n
    assert foo.cols == n
    assert foo.data.dims["rows"] == n
    assert foo.data.dims["cols"] == n
    assert np.array_equal(foo.data.i, arr)
    assert np.array_equal(foo.data.j, arr)


@pytest.mark.xfail(reason="TODO")
def test_dim_coord_with_multiple_coord_aliases():
    @xattree
    class Foo:
        rows: int = dim(coord=("j", "row"))
        cols: int = dim(coord=("i", "col"))

    n = 3
    arr = np.arange(n)
    foo = Foo(rows=n, cols=n)
    assert foo.rows == n
    assert foo.cols == n
    assert foo.layers == n
    assert foo.data.dims["rows"] == n
    assert foo.data.dims["cols"] == n
    assert np.array_equal(foo.data.i, arr)
    assert np.array_equal(foo.data.j, arr)
    assert np.array_equal(foo.data.row, arr)
    assert np.array_equal(foo.data.col, arr)
