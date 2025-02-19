from typing import Optional

import numpy as np
import pytest
from attrs import Factory, define, field
from numpy.typing import NDArray

from xattree import CannotExpand, DimsNotFound, array, dim, xattree


@xattree
class Foo:
    num: int = dim(default=3, coord="n")
    arr: NDArray[np.float64] = array(default=0.0, dims=("num",))


def test_scalar_array_default():
    foo = Foo()

    assert foo.num == 3
    assert np.array_equal(foo.data.n, np.arange(3))
    assert np.array_equal(foo.arr, np.zeros((3)))


def test_scalar_array_accepts_list():
    foo = Foo(arr=[1.0, 1.0, 1.0])

    assert foo.num == 3
    assert np.array_equal(foo.data.n, np.arange(3))
    assert np.array_equal(foo.arr, np.ones((3)))


def test_optional_scalar_array():
    @xattree
    class Foo:
        num: int = dim(coord="n")
        arr: Optional[NDArray[np.float64]] = array(default=None, dims=("num",))

    foo = Foo(num=3)

    assert foo.num == 3
    assert np.array_equal(foo.data.n, np.arange(3))
    assert foo.arr is None


def test_scalar_array_explicit_dims():
    """
    When an array with a scalar default value is not initialized
    but the instance is provided a matching dimensions, the array
    should be expanded to the requested shape.
    """

    @xattree
    class Foo:
        arr: NDArray[np.float64] = array(default=0.0, dims=("num",))

    foo = Foo(dims={"num": 5})

    assert foo.dims["num"] == 5
    assert np.array_equal(foo.data.num, np.arange(5))
    assert np.array_equal(foo.arr, np.zeros((5)))


@xattree
class Baz:
    a: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))


def test_dims_not_found():
    """
    When an array's requested dimension(s) can't be found,
    raise an error by default (because `strict=True`).
    """
    with pytest.raises(
        DimsNotFound, match=r".*failed dim resolution: rows, cols.*"
    ):
        Baz(a=np.arange(3))


@pytest.mark.xfail(reason="TODO: fixme")
def test_no_dims_with_value_relaxed():
    """
    When an array's requested dimension(s) can't be found,
    `strict=False`, and an array value is provided, allow
    the array to be initialized without dim verification.
    """
    a = np.arange(3)
    baz = Baz(a=a, strict=False)
    assert np.array_equal(baz.a, a)
    assert np.array_equal(baz.data.a, a)


def test_no_dims_no_value_relaxed():
    """
    When an array's requested dimension(s) can't be found,
    `strict=False`, and an array value is not provided, do
    not raise an error but add nothing to the `DataTree`
    node's dataset.
    """
    arrs = Baz(strict=False)
    assert not any(arrs.data)


def test_no_dims_expand_fails():
    """
    When no dimensions are specified for an array variable,
    it cannot be expanded from a (scalar) default value, so
    we expand initialization to fail.
    """
    with pytest.raises(CannotExpand, match=r".*no dims, no scalar defaults.*"):

        @xattree
        class Bad:
            a: NDArray[np.float64] = array(default=0.0)


def test_record_array_default():
    @xattree
    class Records:
        @define
        class Record:
            i: int = field(default=0)

        num: int = dim(coord="n")
        arr: NDArray[np.object_] = array(Record, dims=("num",))

    records = Records(num=3)
    assert len(records.arr) == 3
    assert all(isinstance(r, Records.Record) for r in records.arr.to_numpy())


def test_scalar_union_array():
    @xattree
    class Unions:
        num: int = dim(default=3, coord="n")
        arr: NDArray[np.object_] = array(
            np.int64 | np.float64, default=1, dims=("num",)
        )

    unions = Unions()
    assert unions.arr.dtype is np.dtype(np.int64)
    assert np.array_equal(unions.arr, np.ones(3, dtype=np.int64))

    # TODO: type checking that values are members of union? any way to get
    # beartype to do this?
    # arr = np.array([True, True, False])
    # assert pytest.raises(TypeError, Unions, arr=arr)


def test_record_union_array():
    @define
    class RecordA:
        i: int = field(default=0)

    @define
    class RecordB:
        f: float = field(default=0.0)

    @xattree
    class Unions:
        num: int = dim(default=3, coord="n")
        arr: NDArray[np.object_] = array(
            RecordA | RecordB, default=Factory(RecordA), dims=("num",)
        )

    unions = Unions()
    assert unions.arr.dtype is np.dtype(np.object_)
    assert np.array_equal(unions.arr, np.full(3, RecordA()))
