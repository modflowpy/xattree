import numpy as np
import pytest
from attrs import Factory, define, field
from numpy.typing import NDArray

from xattree import CannotExpand, DimsNotFound, array, dim, xattree


@xattree
class Foo:
    num: int = dim(default=10, coord="n")
    arr: NDArray[np.float64] = array(default=0.0, dims=("num",))


def test_scalar_array():
    foo = Foo()

    assert foo.num == 10
    assert np.array_equal(foo.data.n, np.arange(10))
    assert np.array_equal(foo.arr, np.zeros((10)))


@xattree
class Bar:
    arr: NDArray[np.float64] = array(default=0.0, dims=("num",))


def test_scalar_array_explicit_dims():
    """
    When an array with a scalar default value is not initialized
    but the instance is provided a matching dimensions, the array
    should be expanded to the requested shape.
    """
    bar = Bar(dims={"num": 5})

    assert bar.dims["num"] == 5
    assert np.array_equal(bar.data.num, np.arange(5))
    assert np.array_equal(bar.arr, np.zeros((5)))


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


@pytest.mark.xfail(reason="TODO unbork catspec")
def test_object_array_with_default():
    @xattree
    class RecordContainerFactoryDefault:
        @define
        class Record:
            i: int = field(default=0)

        num: int = dim(default=10, coord="n")
        arr: list[Record] = array(default=Factory(Record), dims=("num",))

    ctnr = RecordContainerFactoryDefault()
    assert ctnr.arr == []


@pytest.mark.xfail(reason="TODO default to using factory")
def test_object_array_no_default():
    @xattree
    class RecordContainerNoDefault:
        @define
        class Record:
            i: int = field(default=0)

        num: int = dim(default=10, coord="n")
        arr: list[Record] = array(dims=("num",))

    ctnr = RecordContainerNoDefault()
    assert ctnr.arr == []
