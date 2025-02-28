import attrs
import numpy as np
from numpy.typing import NDArray

from xattree import _Xattribute, array, xattree


def array_from_dict(data):
    if isinstance(data, dict):
        max_index = max(data.keys())
        arr = np.full(max_index + 1, np.nan)
        for k, v in data.items():
            arr[k] = v
        return arr
    return np.array(data)


def test_array_converter():
    @xattree
    class TestClass:
        a: NDArray[np.integer] = array(converter=array_from_dict)

    obj = TestClass(a={0: 1, 2: 3})
    expected = np.array([1.0, np.nan, 3.0])
    np.testing.assert_array_equal(obj.a, expected)


def test_array_converter_takes_self():
    def convert(value, self):
        assert isinstance(self, TestClass)
        return array_from_dict(value) + 1

    @xattree
    class TestClass:
        a: NDArray[np.integer] = array(converter=attrs.Converter(convert, takes_self=True))

    obj = TestClass(a=1)
    expected = np.array(2)
    np.testing.assert_array_equal(obj.a, expected)


def test_array_converter_takes_field():
    def convert(value, field):
        assert isinstance(field, _Xattribute)
        return array_from_dict(value) + 1

    @xattree
    class TestClass:
        a: NDArray[np.integer] = array(converter=attrs.Converter(convert, takes_field=True))

    obj = TestClass(a=1)
    expected = np.array(2)
    np.testing.assert_array_equal(obj.a, expected)


def test_array_converter_takes_self_and_field():
    def convert(value, self, field):
        assert isinstance(self, TestClass)
        assert isinstance(field, _Xattribute)
        return array_from_dict(value) + 1

    @xattree
    class TestClass:
        a: NDArray[np.integer] = array(
            converter=attrs.Converter(convert, takes_self=True, takes_field=True)
        )

    obj = TestClass(a=1)
    expected = np.array(2)
    np.testing.assert_array_equal(obj.a, expected)
