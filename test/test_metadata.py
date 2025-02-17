import numpy as np
import pytest
from attrs import define, field
from numpy.typing import NDArray

from xattree import DimsNotFound, ExpandFailed, xattree


@xattree
@define(slots=False)
class Foo:
    rows: int = field(metadata={"dim": {"coord": "j"}})
    cols: int = field(metadata={"dim": {"coord": "i"}})


def test_dim():
    n = 3
    arr = np.arange(n)
    foo = Foo(rows=n, cols=n)
    assert foo.rows == n
    assert foo.cols == n
    assert foo.data.dims["rows"] == n
    assert foo.data.dims["cols"] == n
    # TODO: attr access for "implicit"
    # coords, i.e., those that are not
    # defined with a top-level "coord"
    # entry in the metadata but inside
    # a "dim" entry.
    # assert np.array_equal(foo.i, arr)
    # assert np.array_equal(foo.j, arr)
    assert np.array_equal(foo.data.i, arr)
    assert np.array_equal(foo.data.j, arr)


@xattree
@define(slots=False)
class Bar:
    i: NDArray[np.int64] = field(metadata={"coord": {"dim": "cols"}})
    j: NDArray[np.int64] = field(metadata={"coord": {"dim": "rows"}})


def test_coord():
    n = 3
    arr = np.arange(n)
    bar = Bar(i=arr, j=arr)
    # TODO: attr access for "implicit"
    # dims, i.e., those that are not
    # defined with a top-level "dim"
    # entry in metadata but inside
    # a "coord" entry.
    # assert bar.rows == n
    # assert bar.cols == n
    assert bar.data.dims["rows"] == n
    assert bar.data.dims["cols"] == n
    assert np.array_equal(bar.i, arr)
    assert np.array_equal(bar.j, arr)
    assert np.array_equal(bar.data.i, arr)
    assert np.array_equal(bar.data.j, arr)


@xattree
@define(slots=False)
class Baz:
    a: NDArray[np.float64] = field(
        default=0.0, metadata={"dims": ("rows", "cols")}
    )


def test_dims_not_found_strict():
    """
    When an array's requested dimension(s) can't be found
    and `strict=True`, an error should be raised.
    """
    with pytest.raises(
        DimsNotFound, match=r".*failed dim resolution: rows, cols.*"
    ):
        Baz(a=np.arange(3), strict=True)


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


@xattree
@define(slots=False)
class Bad:
    a: NDArray[np.float64] = field(default=0.0)


@pytest.mark.parametrize("strict", [True, False])
def test_no_dims_expand_fails(strict):
    """
    When no dimensions are specified for an array variable,
    it cannot be expanded from a (scalar) default value, so
    we expand initialization to fail regardless of `strict`.
    """
    with pytest.raises(ExpandFailed, match=r".*can't expand, no dims.*"):
        Bad(strict=strict)
