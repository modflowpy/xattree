import attrs
import pytest

from xattree import field, xattree


@xattree
class UnsafeFoo:
    s: str = field()
    i: int = field()


@xattree
class Foo:
    s: str = field(validator=attrs.validators.instance_of(str))
    i: int = field(validator=attrs.validators.instance_of(int))


def test_validators():
    UnsafeFoo(s=123, i=30)

    example = Foo(s="hello", i=30)
    assert example.s == "hello"
    assert example.i == 30

    with pytest.raises(TypeError):
        Foo(s=123, i=30)

    with pytest.raises(TypeError):
        Foo(s="John", i="30")
