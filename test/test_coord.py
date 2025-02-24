import numpy as np
from numpy.typing import NDArray

from xattree import coord, xattree


@xattree
class Foo:
    row: NDArray[np.int64] = coord()
    col: NDArray[np.int64] = coord()


def test_coord():
    n = 3
    arr = np.arange(n)
    foo = Foo(row=arr, col=arr)
    assert foo.data.dims["row"] == n
    assert foo.data.dims["col"] == n
    assert np.array_equal(foo.row, arr)
    assert np.array_equal(foo.col, arr)
    assert np.array_equal(foo.data.row, arr)
    assert np.array_equal(foo.data.col, arr)
