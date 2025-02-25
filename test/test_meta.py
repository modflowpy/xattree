import attrs
import numpy as np
from numpy.typing import NDArray

from xattree import (
    _get_xatspec,
    array,
    coord,
    dim,
    field,
    fields,
    fields_dict,
    has_xats,
    is_xat,
    xattree,
)


@xattree
class Foo:
    i: int = field()
    d: int = dim()
    n: int = attrs.field()


class Bar:
    pass


def test_has_xats():
    assert has_xats(Foo)
    assert not has_xats(Bar)


def test_is_xat():
    fields_ = fields_dict(Foo)
    assert is_xat(fields_["i"])
    assert is_xat(fields_["d"])
    assert not is_xat(fields_["n"])


def test_fields_just_yours():
    fields_ = fields(Foo)
    assert len(fields_) == 3
    assert fields_[0].name == "i"
    assert fields_[1].name == "d"
    assert fields_[2].name == "n"
    assert list(fields_dict(Foo).values()) == fields_


def test_fields_all():
    fields_ = fields(Foo, just_yours=False)
    assert len(fields_) == 7
    assert fields_[0].name == "i"
    assert fields_[1].name == "d"
    assert fields_[2].name == "n"
    assert fields_[3].name == "name"
    assert fields_[4].name == "dims"
    assert fields_[5].name == "parent"
    assert fields_[6].name == "strict"
    assert list(fields_dict(Foo, just_yours=False).values()) == fields_


def test_xatspec_simple():
    @xattree
    class Foo:
        c: NDArray[np.integer] = coord()
        a: NDArray[np.floating] = array()

    xatspec = _get_xatspec(Foo)
    assert "c" in xatspec.coords
    assert "a" in xatspec.arrays
    c = xatspec.coords["c"]
    a = xatspec.arrays["a"]
    assert c.name == "c"
    assert a.name == "a"
    assert c.scope is None
    assert c.path is None
