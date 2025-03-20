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
