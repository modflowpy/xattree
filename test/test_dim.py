from typing import Optional

import numpy as np

from xattree import dim, xattree


@xattree
class Foo:
    rows: int = dim(coord="j")
    cols: int = dim(coord="i")
    layers: Optional[int] = dim(coord="k", default=None)


def test_dim():
    n = 3
    arr = np.arange(n)
    foo = Foo(rows=n, cols=n)
    assert foo.rows == n
    assert foo.cols == n
    assert foo.data.dims["rows"] == n
    assert foo.data.dims["cols"] == n
    assert np.array_equal(foo.data.i, arr)
    assert np.array_equal(foo.data.j, arr)
