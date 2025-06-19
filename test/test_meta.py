import attrs
import numpy as np
from numpy.typing import NDArray

from xattree import (
    array,
    coord,
    dim,
    field,
    fields,
    fields_dict,
    get_xatspec,
    has,
    is_xat,
    xattree,
)


@xattree
class Foo:
    @attrs.define
    class Bar:
        pass

    i: int = field()
    d: int = dim()
    n: int = attrs.field()
    c: Bar = attrs.field()


class Bar:
    pass


def test_has_xats():
    assert has(Foo)
    assert not has(Bar)


def test_is_xat():
    fields_ = fields_dict(Foo)
    assert is_xat(fields_["i"])
    assert is_xat(fields_["d"])
    assert not is_xat(fields_["n"])
    assert not is_xat(fields_["c"])


def test_fields():
    fields_ = fields(Foo)
    assert len(fields_) == 4
    assert fields_[0].name == "i"
    assert fields_[1].name == "d"
    assert fields_[2].name == "n"
    assert list(fields_dict(Foo).values()) == fields_


def test_fields_extra():
    fields_ = fields(Foo, extra=True)
    assert len(fields_) == 10
    assert fields_[0].name == "i"
    assert fields_[1].name == "d"
    assert fields_[2].name == "n"
    assert fields_[3].name == "c"
    assert fields_[4].name == "name"
    assert fields_[5].name == "dims"
    assert fields_[6].name == "parent"
    assert fields_[7].name == "children"
    assert fields_[8].name == "strict"
    assert fields_[9].name == "data"
    assert list(fields_dict(Foo, extra=True).values()) == fields_


def test_xatspec():
    @xattree
    class Foo:
        c: NDArray[np.integer] = coord()
        a: NDArray[np.floating] = array()

    xatspec = get_xatspec(Foo)
    assert "c" in xatspec.coords
    assert "a" in xatspec.arrays
    c = xatspec.coords["c"]
    a = xatspec.arrays["a"]
    assert c.name == "c"
    assert a.name == "a"
    assert c.scope is None
    assert c.path is None
