import attrs
import numpy as np
import pytest
from numpy.typing import NDArray

from xattree import array, field, xattree


def test_scalar_fields_with_auto_type_validation():
    @xattree
    class Foo:
        s: str = field()
        i: int = field()

    foo = Foo(s="hello", i=30)
    assert foo.s == "hello"
    assert foo.i == 30

    with pytest.raises(TypeError):
        Foo(s=123, i=30)

    with pytest.raises(TypeError):
        Foo(s="hello", i="30")


def test_scalar_fields_with_explicit_validators():
    @xattree
    class Foo:
        s: str = field(validator=attrs.validators.instance_of(str))
        i: int = field(validator=attrs.validators.instance_of(int))

    foo = Foo(s="hello", i=30)
    assert foo.s == "hello"
    assert foo.i == 30

    with pytest.raises(TypeError):
        Foo(s=123, i=30)

    with pytest.raises(TypeError):
        Foo(s="hello", i="30")


@xattree
class ArrayFoo:
    arr: NDArray[np.integer] = array(
        validator=[
            attrs.validators.instance_of(np.ndarray),
            attrs.validators.deep_iterable(
                member_validator=attrs.validators.instance_of(np.integer),
                iterable_validator=attrs.validators.instance_of(np.ndarray),
            ),
        ]
    )


def test_array_validators():
    valid_array = np.array([1, 2, 3])
    example = ArrayFoo(arr=valid_array)
    assert np.array_equal(example.arr, valid_array)

    with pytest.raises(TypeError):
        ArrayFoo(arr="not an array")

    with pytest.raises(TypeError):
        ArrayFoo(arr=np.array(["a", "b", "c"]))

    with pytest.raises(TypeError):
        ArrayFoo(arr=np.array([1.0, 2.0, 3.0]))
