import pytest
from attrs import define, field
from xarray import DataTree

from xattree import xattree


@xattree
@define(slots=False)
class Foo:
    a: int = field()
    b: int = field(default=42)
    c: float = field(default=1.0)


def test_required_attrs():
    """
    If a required field is not initialized, an error should be raised.
    """
    with pytest.raises(
        TypeError, match="missing 1 required positional argument"
    ):
        Foo()


def test_dict_empty():
    """
    `attrs` fields should not be stored in `__dict__`, even
    though `slots=False`. The instance `__dict__` should be
    empty except for an `xarray.DataTree` under key "data".
    """
    foo = Foo(a=0)
    assert set(foo.__dict__.keys()) == {"data"}
    assert isinstance(foo.__dict__["data"], DataTree)
    assert foo.data is foo.__dict__["data"]


def test_access():
    """`attrs` attribute access should still work as expected."""
    foo = Foo(a=0)
    assert foo.a == 0
    assert foo.b == 42
    assert foo.c == 1.0


@pytest.mark.xfail(reason="TODO implement mutations")
def test_mutate():
    """
    `attrs` class attributes should be mutable, with all
    mutations reflected in the data tree and vice versa.
    """

    foo = Foo(a=3)
    assert foo.a == 3
    foo.a = 4
    assert foo.a == 4
    assert foo.data.attrs["a"] == 4
    foo.data.attrs["a"] = 5
    assert foo.a == 5
    assert set(foo.__dict__.keys()) == {"data"}
