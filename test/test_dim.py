import numpy as np
from numpy.typing import NDArray

from xattree import array, dim, xattree


def test_dim_coord():
    """
    By default, an unmodified dimension field becomes a dimension
    coordinate. An eponymous coordinate array is created with the
    same name as the field.
    """

    @xattree
    class Foo:
        t: int = dim()

    n = 3
    foo = Foo(t=n)
    assert foo.t == n
    assert foo.data.dims["t"] == n
    assert "t" in foo.data.coords
    assert "t" not in foo.data.attrs
    assert np.array_equal(foo.data.t, np.arange(n))


def test_dim_without_coord():
    """
    A dimension can be created without a coordinate array by
    setting the `coord` argument to `False`. If any arrays
    are registered with the dimension, the dimension will be
    present in the `data.dims` and `data.attrs` dictionaries.
    If no arrays are registered, the dimension will only be
    present in the `data.attrs` dictionary.
    """

    @xattree
    class Foo:
        t: int = dim(coord=False)

    n = 3
    foo = Foo(t=n)
    assert foo.t == n

    @xattree
    class Bar:
        t: int = dim(coord=False)
        a: NDArray = array(default=0.0, dims=("t",))

    n = 3
    bar = Bar(t=n)
    assert bar.t == n
    assert bar.data.dims["t"] == n
    assert bar.data.attrs["t"] == n
    assert "t" not in bar.data.coords


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
