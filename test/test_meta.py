from attrs import field

from xattree import fields, fields_dict, xats, xattree


@xattree
class Foo:
    i: int = field()


class Bar:
    pass


def test_xat():
    assert xats(Foo)
    assert not xats(Bar)


def test_fields_just_yours():
    fields_ = fields(Foo)
    assert len(fields_) == 1
    assert fields_[0].name == "i"
    assert list(fields_dict(Foo).values()) == fields_


def test_fields_with_xattrs():
    fields_ = fields(Foo, xattrs=True)
    assert len(fields_) == 4
    assert fields_[0].name == "i"
    assert fields_[1].name == "name"
    assert fields_[2].name == "parent"
    assert fields_[3].name == "strict"
    assert list(fields_dict(Foo, xattrs=True).values()) == fields_
