"""A super simple case."""

from pathlib import Path
from typing import Optional

import pytest
from attrs import Factory
from xarray import DataTree

from xattree import _get_xatspec, field, xattree


@xattree
class Foo:
    a: int = field()
    b: int = field(default=42)
    c: float = field(default=1.0)
    p: Path = field(default=Factory(Path.cwd))
    op: Optional[Path] = field(default=None, metadata={"block": "options"})


def test_meta():
    spec = _get_xatspec(Foo).flat
    assert "a" in spec
    assert "b" in spec
    assert "c" in spec
    assert "p" in spec
    assert "op" in spec


def test_required_attrs():
    """
    If a required field is not initialized, an error should be raised.
    """
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        Foo()


def test_dict_empty():
    """
    `attrs` fields should not be stored in `__dict__`, even
    though `slots=False`. The instance `__dict__` should be
    empty except for some entries required by the cat-tree.
    """
    foo = Foo(a=0)
    assert set(foo.__dict__.keys()) == {
        "data",
        "strict",
        "_xattree_ready",
    }
    assert isinstance(foo.__dict__["data"], DataTree)
    assert foo.data is foo.__dict__["data"]


def test_access():
    """`attrs` attribute access should still work as expected."""
    foo = Foo(a=0)
    assert foo.a == 0
    assert foo.b == 42
    assert foo.c == 1.0
    assert foo.p == Path.cwd()


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
    assert set(foo.__dict__.keys()) == {
        "data",
        "strict",
        "_xattree_ready",
    }
