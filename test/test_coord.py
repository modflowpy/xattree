import numpy as np
from numpy.typing import NDArray

from xattree import coord, xattree


@xattree
class Foo:
    i: NDArray[np.int64] = coord(dim="cols")
    j: NDArray[np.int64] = coord(dim="rows")


def test_coord():
    n = 3
    arr = np.arange(n)
    foo = Foo(i=arr, j=arr)
    assert foo.data.dims["rows"] == n
    assert foo.data.dims["cols"] == n
    assert np.array_equal(foo.i, arr)
    assert np.array_equal(foo.j, arr)
    assert np.array_equal(foo.data.i, arr)
    assert np.array_equal(foo.data.j, arr)
