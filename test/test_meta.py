from attrs import field

from xattree import fields, fields_dict, has_xats, xattree


@xattree
class Foo:
    i: int = field()


class Bar:
    pass


def test_xat():
    assert has_xats(Foo)
    assert not has_xats(Bar)


def test_fields_just_yours():
    fields_ = fields(Foo)
    assert len(fields_) == 1
    assert fields_[0].name == "i"
    assert list(fields_dict(Foo).values()) == fields_


def test_fields_all():
    fields_ = fields(Foo, just_yours=False)
    assert len(fields_) == 5
    assert fields_[0].name == "i"
    assert fields_[1].name == "name"
    assert fields_[2].name == "dims"
    assert fields_[3].name == "parent"
    assert fields_[4].name == "strict"
    assert list(fields_dict(Foo, just_yours=False).values()) == fields_
