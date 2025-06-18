import numpy as np
import pytest
from numpy.typing import NDArray

from xattree import array, dim, sparse_dict_converter, xattree


def test_sparse_dict_converter_just_grouped_dims():
    @xattree
    class Foo:
        t: int = dim(group="time")
        x: int = dim(group="space")
        y: int = dim(group="space")
        a: NDArray[np.float64] = array(
            dims=("t", "x", "y"),
            converter=sparse_dict_converter,
        )

    # time -> (lat, lon) -> value
    d = {0: {(0, 0): 25.5, (1, 1): 26.0}, 2: {(0, 1): 23.0}}

    foo = Foo(t=3, x=2, y=2, a=d)

    assert foo.a[0, 0, 0] == 25.5
    assert foo.a[0, 1, 1] == 26.0
    assert foo.a[2, 0, 1] == 23.0
    assert np.isnan(foo.a[1, 0, 0])
    assert np.isnan(foo.a[0, 0, 1])


def test_sparse_dict_converter_grouped_and_ungrouped_dims():
    @xattree
    class Foo:
        t: int = dim(group="time")
        x: int = dim(group="space")
        y: int = dim(group="space")
        z: int = dim()

        a: NDArray[np.float64] = array(
            dims=("t", "x", "y", "z"),
            converter=sparse_dict_converter,
        )

    # time -> (lat, lon) -> depth -> value
    d = {0: {(0, 0): {0: 25.5, 2: 24.0}, (1, 1): {1: 26.0}}, 1: {(0, 1): {0: 23.0, 1: 22.5}}}

    foo = Foo(t=2, x=2, y=2, z=3, a=d)

    assert foo.a[0, 0, 0, 0] == 25.5
    assert foo.a[0, 0, 0, 2] == 24.0
    assert foo.a[0, 1, 1, 1] == 26.0
    assert foo.a[1, 0, 1, 0] == 23.0
    assert foo.a[1, 0, 1, 1] == 22.5
    assert np.isnan(foo.a[0, 0, 0, 1])
    assert np.isnan(foo.a[1, 1, 1, 1])


def test_sparse_dict_converter_invalid_coords():
    @xattree
    class Foo:
        t: int = dim(group="time")
        x: int = dim(group="space")
        y: int = dim(group="space")
        a: NDArray[np.float64] = array(
            dims=("t", "x", "y"),
            converter=sparse_dict_converter,
        )

    with pytest.raises(ValueError, match="expected tuple of 2 coordinates"):
        Foo(t=2, x=2, y=2, a={0: {(1,): 25.0}})

    with pytest.raises(ValueError, match="expected tuple of 2 coordinates"):
        Foo(t=2, x=2, y=2, a={0: {1: 25.0}})


def test_sparse_dict_converter_fill_value_from_default():
    """Test that scalar defaults are used as fill values."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        a: NDArray[np.int32] = array(
            dims=("t", "x"),
            converter=sparse_dict_converter,
            default=-999,  # Scalar default should be used as fill value
        )

    d = {0: {0: 42}}
    foo = Foo(t=2, x=2, a=d)

    assert foo.a[0, 0].item() == 42
    assert foo.a[0, 1].item() == -999
    assert foo.a[1, 0].item() == -999
    assert foo.a[1, 1].item() == -999


def test_sparse_dict_converter_fill_value_dtype_inference():
    """Test that fill values are inferred from dtype when no default."""

    # Integer array should get 0 fill
    @xattree
    class IntFoo:
        t: int = dim()
        x: int = dim()
        a: NDArray[np.int32] = array(
            dims=("t", "x"),
            converter=sparse_dict_converter,
        )

    d = {0: {0: 42}}
    foo = IntFoo(t=2, x=2, a=d)
    assert foo.a[0, 0].item() == 42
    assert foo.a[0, 1].item() == 0
    assert foo.a[1, 0].item() == 0

    # Float array should get NaN fill
    @xattree
    class FloatFoo:
        t: int = dim()
        x: int = dim()
        a: NDArray[np.float64] = array(
            dims=("t", "x"),
            converter=sparse_dict_converter,
        )

    d = {0: {0: 3.14}}
    foo = FloatFoo(t=2, x=2, a=d)
    assert foo.a[0, 0].item() == 3.14
    assert np.isnan(foo.a[0, 1].item())
    assert np.isnan(foo.a[1, 0].item())

    # Object array should get None fill
    @xattree
    class ObjFoo:
        t: int = dim()
        x: int = dim()
        a: NDArray[np.object_] = array(
            dims=("t", "x"),
            converter=sparse_dict_converter,
        )

    d = {0: {0: "hello"}}
    foo = ObjFoo(t=2, x=2, a=d)
    assert foo.a[0, 0].item() == "hello"
    assert foo.a[0, 1].item() is None
    assert foo.a[1, 0].item() is None


def test_sparse_dict_converter_fill_value_bool_dtype():
    """Test bool dtype gets False fill value."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        a: NDArray[np.bool_] = array(
            dims=("t", "x"),
            converter=sparse_dict_converter,
        )

    d = {0: {0: True}}
    foo = Foo(t=2, x=2, a=d)
    assert foo.a[0, 0].item()
    assert not foo.a[0, 1].item()
    assert not foo.a[1, 0].item()


def test_sparse_dict_converter_fill_value_string_dtype():
    """Test string dtype gets empty string fill value."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        a: NDArray[np.str_] = array(
            dims=("t", "x"),
            converter=sparse_dict_converter,
        )

    d = {0: {0: "hello"}}
    foo = Foo(t=2, x=2, a=d)
    assert foo.a[0, 0].item() == "hello"
    assert foo.a[0, 1].item() == ""
    assert foo.a[1, 0].item() == ""


def test_sparse_dict_converter_fill_value_complex_dtype():
    """Test complex dtype gets complex NaN fill value."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        a: NDArray[np.complex128] = array(
            dims=("t", "x"),
            converter=sparse_dict_converter,
        )

    d = {0: {0: 1 + 2j}}
    foo = Foo(t=2, x=2, a=d)
    assert foo.a[0, 0].item() == 1 + 2j
    # Both real and imaginary parts should be NaN
    assert np.isnan(foo.a[0, 1].item().real)
    assert np.isnan(foo.a[0, 1].item().imag)


def test_sparse_dict_converter_default_over_dtype():
    """Test that scalar default takes precedence over dtype inference."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        a: NDArray[np.int32] = array(
            dims=("t", "x"),
            converter=sparse_dict_converter,
            default=42,  # Scalar default should be used instead of 0
        )

    d = {0: {0: 100}}
    foo = Foo(t=2, x=2, a=d)
    assert foo.a[0, 0].item() == 100
    assert foo.a[0, 1].item() == 42  # Default used, not 0
    assert foo.a[1, 0].item() == 42


def test_sparse_dict_converter_non_scalar_default_ignored():
    """Test that non-scalar defaults are ignored for fill value."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        a: NDArray[np.int32] = array(
            dims=("t", "x"),
            converter=sparse_dict_converter,
            default=np.array([1, 2, 3]),  # Non-scalar, should be ignored
        )

    d = {0: {0: 100}}
    foo = Foo(t=2, x=2, a=d)
    assert foo.a[0, 0].item() == 100
    assert foo.a[0, 1].item() == 0  # Dtype inference used (0 for int)
    assert foo.a[1, 0].item() == 0


def test_sparse_dict_converter_passthrough_unchanged():
    """Test that non-dict values are passed through unchanged."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        a: NDArray[np.float64] = array(
            dims=("t", "x"),
            converter=sparse_dict_converter,
        )

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    foo = Foo(t=2, x=2, a=arr)

    np.testing.assert_array_equal(foo.a, arr)
