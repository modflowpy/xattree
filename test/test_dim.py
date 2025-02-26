import numpy as np

from xattree import dim, xattree


def test_dim_coord():
    """
    By default, an unmodified dimension field becomes a dimension
    coordinate. An eponymous coordinate array is created with the
    same name as the field. Attribute access returns the dim size.
    The coordinate array is accessible only through the data tree.
    Dimension size is stored as a data tree attribute.
    """

    @xattree
    class Foo:
        t: int = dim()

    n = 3
    foo = Foo(t=n)
    assert foo.t == n
    assert foo.data.dims["t"] == n
    assert foo.data.attrs["t"] == n
    assert np.array_equal(foo.data.t, np.arange(n))


def test_dim_coord_aliased():
    """
    Test a dimension coordinate with an alias*. This is useful
    if one wants a different name for an xarray dimension and
    coordinate array than one wants for an object model. This
    is like a normal dimension in that the coordinate array is
    only accessible through the data tree. The dimension size
    is stored under the original field name in the data tree's
    attrs, while the dimension and coordinate array are stored
    under the alias.

    *Note that this is not the same as an attrs alias, which
    is a way to reassign a field's `__init__ parameter name.
    """

    @xattree
    class Foo:
        rows: int = dim(name="row")
        cols: int = dim(name="col")

    n = 3
    arr = np.arange(n)
    foo = Foo(rows=n, cols=n)
    assert foo.rows == n
    assert foo.cols == n
    assert foo.data.attrs["rows"] == n
    assert foo.data.attrs["cols"] == n
    assert foo.data.dims["row"] == n
    assert foo.data.dims["col"] == n
    assert np.array_equal(foo.data.row, arr)
    assert np.array_equal(foo.data.col, arr)


def test_derived_dim():
    """
    A derived dimension is a dimension that is computed from
    other dimensions in the `__attrs_post_init__` hook, also
    recognizable by using `init=False` in the dim decorator.
    """

    @xattree
    class Foo:
        rows: int = dim()
        cols: int = dim()
        nodes: int = dim(init=False)

        def __attrs_post_init__(self):
            self.nodes = self.rows * self.cols

    n = 3
    foo = Foo(rows=n, cols=n)
    nodes = n * n
    assert foo.nodes == nodes
    assert foo.data.dims["nodes"] == nodes
    assert foo.data.attrs["nodes"] == nodes
