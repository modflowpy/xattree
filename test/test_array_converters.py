import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from xattree import array, dict_to_array_converter, dim, table_converter, xattree


def test_sparse_dict_converter_just_grouped_dims():
    @xattree
    class Foo:
        t: int = dim(group="time")
        x: int = dim(group="space")
        y: int = dim(group="space")
        a: NDArray[np.float64] = array(
            dims=("t", "x", "y"),
            converter=dict_to_array_converter,
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
            converter=dict_to_array_converter,
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
            converter=dict_to_array_converter,
        )

    with pytest.raises(ValueError, match="Expected tuple of 2 coords"):
        Foo(t=2, x=2, y=2, a={0: {(1,): 25.0}})

    with pytest.raises(ValueError, match="Expected tuple of 2 coords"):
        Foo(t=2, x=2, y=2, a={0: {1: 25.0}})


def test_sparse_dict_converter_fill_value_from_default():
    """Test that scalar defaults are used as fill values."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        a: NDArray[np.int32] = array(
            dims=("t", "x"),
            converter=dict_to_array_converter,
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
            converter=dict_to_array_converter,
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
            converter=dict_to_array_converter,
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
            converter=dict_to_array_converter,
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
            converter=dict_to_array_converter,
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
            converter=dict_to_array_converter,
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
            converter=dict_to_array_converter,
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
            converter=dict_to_array_converter,
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
            converter=dict_to_array_converter,
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
            converter=dict_to_array_converter,
        )

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    foo = Foo(t=2, x=2, a=arr)

    np.testing.assert_array_equal(foo.a, arr)


def test_sparse_dict_converter_inherited_dims():
    """Test sparse dict converter with dimensions defined in parent components."""
    from xattree import ROOT, field

    @xattree
    class Grid:
        t: int = dim(scope=ROOT)
        x: int = dim(scope=ROOT)
        y: int = dim(scope=ROOT)

    @xattree
    class Arrs:
        a: NDArray[np.float64] = array(
            dims=("t", "x", "y"),
            converter=dict_to_array_converter,
            default=np.nan,
        )

    @xattree
    class Root:
        grid: Grid = field()
        arrs: Arrs = field()

    grid = Grid(t=3, x=2, y=2)
    root = Root(grid=grid)
    data = Arrs(
        parent=root,
        a={0: {0: {0: 10.0, 1: 20.0}, 1: {0: 30.0, 1: 40.0}}, 2: {0: {1: 50.0}, 1: {0: 60.0}}},
    )

    assert data.a.shape == (3, 2, 2)
    assert data.a[0, 0, 0] == 10.0
    assert data.a[0, 0, 1] == 20.0
    assert data.a[0, 1, 0] == 30.0
    assert data.a[0, 1, 1] == 40.0
    assert data.a[2, 0, 1] == 50.0
    assert data.a[2, 1, 0] == 60.0
    assert np.isnan(data.a[1, 0, 0])
    assert np.isnan(data.a[2, 0, 0])


def test_sparse_dict_converter_inherited_grouped_dims():
    from xattree import ROOT, field

    @xattree
    class Grid:
        t: int = dim(scope=ROOT, group="time")
        x: int = dim(scope=ROOT, group="space")
        y: int = dim(scope=ROOT, group="space")

    @xattree
    class Arrs:
        a: NDArray[np.float64] = array(
            dims=("t", "x", "y"),
            converter=dict_to_array_converter,
            default=np.nan,
        )

    @xattree
    class Root:
        grid: Grid = field()
        arrs: Arrs = field()

    grid = Grid(t=3, x=2, y=2)
    root = Root(grid=grid)
    # t -> (x, y) -> value structure since x,y are grouped
    data = Arrs(
        parent=root,
        a={0: {(0, 0): 10.0, (1, 1): 20.0}, 2: {(0, 1): 30.0, (1, 0): 40.0}},
    )

    assert data.a.shape == (3, 2, 2)
    assert data.a[0, 0, 0] == 10.0
    assert data.a[0, 1, 1] == 20.0
    assert data.a[2, 0, 1] == 30.0
    assert data.a[2, 1, 0] == 40.0
    assert np.isnan(data.a[1, 0, 0])
    assert np.isnan(data.a[0, 0, 1])
    assert np.isnan(data.a[2, 0, 0])


def test_table_converter_pandas_dataframe():
    """Test table converter with pandas DataFrame input."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        y: int = dim()
        temp: NDArray[np.float64] = array(
            dims=("t", "x", "y"),
            converter=table_converter,
        )

    df = pd.DataFrame(
        {"t": [0, 0, 1, 1], "x": [0, 1, 0, 1], "y": [0, 0, 1, 1], "temp": [25.3, 26.1, 23.8, 24.5]}
    )

    foo = Foo(t=2, x=2, y=2, temp=df)

    assert foo.temp[0, 0, 0] == 25.3
    assert foo.temp[0, 1, 0] == 26.1
    assert foo.temp[1, 0, 1] == 23.8
    assert foo.temp[1, 1, 1] == 24.5
    # Missing combinations should be NaN (default for float)
    assert np.isnan(foo.temp[0, 0, 1])
    assert np.isnan(foo.temp[1, 0, 0])


def test_table_converter_numpy_recarray():
    """Test table converter with numpy recarray input."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        temp: NDArray[np.float64] = array(
            dims=("t", "x"),
            converter=table_converter,
        )

    # Create a numpy recarray
    data = np.array(
        [(0, 0, 25.3), (0, 1, 26.1), (1, 0, 23.8)], dtype=[("t", "i4"), ("x", "i4"), ("temp", "f8")]
    )

    foo = Foo(t=2, x=2, temp=data)

    assert foo.temp[0, 0] == 25.3
    assert foo.temp[0, 1] == 26.1
    assert foo.temp[1, 0] == 23.8
    assert np.isnan(foo.temp[1, 1])  # Missing combination


@pytest.mark.skip(reason="TODO")
def test_table_converter_multiple_value_columns():
    """Test table converter with multiple value columns creating record objects."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        measurements: NDArray[np.object_] = array(
            dims=("t", "x"),
            converter=table_converter,
        )

    df = pd.DataFrame(
        {"t": [0, 0, 1], "x": [0, 1, 0], "temp": [25.3, 26.1, 23.8], "humidity": [60.0, 65.0, 55.0]}
    )

    foo = Foo(t=2, x=2, measurements=df)

    # Check record objects are created
    record_00 = foo.measurements[0, 0]
    assert hasattr(record_00, "temp")
    assert hasattr(record_00, "humidity")
    assert record_00.temp == 25.3
    assert record_00.humidity == 60.0

    record_01 = foo.measurements[0, 1]
    assert record_01.temp == 26.1
    assert record_01.humidity == 65.0

    record_10 = foo.measurements[1, 0]
    assert record_10.temp == 23.8
    assert record_10.humidity == 55.0

    # Missing combination should be None
    assert foo.measurements[1, 1] is None


def test_table_converter_partial_coordinate_columns():
    """Test behavior when not all dimensions have corresponding columns."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        y: int = dim()
        temp: NDArray[np.float64] = array(
            dims=("t", "x", "y"),
            converter=table_converter,
        )

    # Only t and x columns, missing y
    df = pd.DataFrame({"t": [0, 1], "x": [0, 1], "temp": [25.3, 26.1]})

    with pytest.raises(ValueError, match="No coordinate columns found matching dimensions"):
        Foo(t=2, x=2, y=2, temp=df)


def test_table_converter_no_value_columns():
    """Test error when no value columns are found."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        temp: NDArray[np.float64] = array(
            dims=("t", "x"),
            converter=table_converter,
        )

    # Only coordinate columns, no value columns
    df = pd.DataFrame({"t": [0, 1], "x": [0, 1]})

    with pytest.raises(ValueError, match="No value columns found in tabular data"):
        Foo(t=2, x=2, temp=df)


def test_table_converter_fill_value_strategies():
    """Test different fill value strategies based on dtype."""

    # Integer array with scalar default
    @xattree
    class IntFoo:
        t: int = dim()
        x: int = dim()
        count: NDArray[np.int32] = array(
            dims=("t", "x"),
            converter=table_converter,
            default=-999,
        )

    df_int = pd.DataFrame({"t": [0], "x": [0], "count": [42]})
    foo_int = IntFoo(t=2, x=2, count=df_int)
    assert foo_int.count[0, 0] == 42
    assert foo_int.count[0, 1] == -999  # Uses scalar default
    assert foo_int.count[1, 0] == -999

    # Float array without default (should use NaN)
    @xattree
    class FloatFoo:
        t: int = dim()
        x: int = dim()
        value: NDArray[np.float64] = array(
            dims=("t", "x"),
            converter=table_converter,
        )

    df_float = pd.DataFrame({"t": [0], "x": [0], "value": [3.14]})
    foo_float = FloatFoo(t=2, x=2, value=df_float)
    assert foo_float.value[0, 0] == 3.14
    assert np.isnan(foo_float.value[0, 1])
    assert np.isnan(foo_float.value[1, 0])


def test_table_converter_coordinate_indexing():
    """Test that coordinates are properly indexed regardless of order in table."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        temp: NDArray[np.float64] = array(
            dims=("t", "x"),
            converter=table_converter,
        )

    # Data not in sorted order
    df = pd.DataFrame({"t": [1, 0, 1, 0], "x": [1, 0, 0, 1], "temp": [20.0, 25.0, 30.0, 35.0]})

    foo = Foo(t=2, x=2, temp=df)

    # Should be indexed by coordinate values, not row order
    assert foo.temp[0, 0] == 25.0  # t=0, x=0
    assert foo.temp[0, 1] == 35.0  # t=0, x=1
    assert foo.temp[1, 0] == 30.0  # t=1, x=0
    assert foo.temp[1, 1] == 20.0  # t=1, x=1


def test_table_converter_empty_table():
    """Test behavior with empty tables when field is optional."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        temp: NDArray[np.float64] | None = array(
            dims=("t", "x"),
            converter=table_converter,
            default=None,
        )

    # Empty DataFrame
    df = pd.DataFrame(columns=["t", "x", "temp"])

    foo = Foo(t=2, x=2, temp=df)
    assert foo.temp is None


def test_table_converter_unsupported_input():
    """Test error handling for unsupported input types."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        temp: NDArray[np.float64] = array(
            dims=("t", "x"),
            converter=table_converter,
        )

    # List (not tabular) should be converted to array normally
    foo = Foo(t=2, x=2, temp=[[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(foo.temp, expected)


def test_table_converter_inherited_dims():
    """Test table converter with dimensions defined in parent components."""
    from xattree import ROOT, field

    @xattree
    class Grid:
        t: int = dim(scope=ROOT)
        x: int = dim(scope=ROOT)
        y: int = dim(scope=ROOT)

    @xattree
    class Arrs:
        a: NDArray[np.float64] = array(
            dims=("t", "x", "y"),
            converter=table_converter,
            default=np.nan,
        )

    @xattree
    class Root:
        grid: Grid = field()
        arrs: Arrs = field()

    grid = Grid(t=2, x=2, y=2)
    root = Root(grid=grid)

    df = pd.DataFrame(
        {"t": [0, 0, 1, 1], "x": [0, 1, 0, 1], "y": [0, 0, 1, 1], "a": [10.0, 20.0, 30.0, 40.0]}
    )

    data = Arrs(parent=root, a=df)

    assert data.a.shape == (2, 2, 2)
    assert data.a[0, 0, 0] == 10.0
    assert data.a[0, 1, 0] == 20.0
    assert data.a[1, 0, 1] == 30.0
    assert data.a[1, 1, 1] == 40.0
    assert np.isnan(data.a[0, 0, 1])
    assert np.isnan(data.a[1, 0, 0])


def test_table_converter_dimension_order_preservation():
    """Test that dimension order from field.dims is preserved."""

    @xattree
    class Foo:
        y: int = dim()  # Note: y comes before x in field order
        t: int = dim()
        x: int = dim()
        temp: NDArray[np.float64] = array(
            dims=("y", "t", "x"),  # Different order than typical
            converter=table_converter,
        )

    # Column order in DataFrame doesn't matter
    df = pd.DataFrame(
        {"x": [0, 1, 0, 1], "t": [0, 0, 1, 1], "y": [0, 0, 1, 1], "temp": [10.0, 20.0, 30.0, 40.0]}
    )

    foo = Foo(y=2, t=2, x=2, temp=df)

    # Shape should follow dims order: (y, t, x)
    assert foo.temp.shape == (2, 2, 2)
    # Access should use dims order: [y, t, x]
    assert foo.temp[0, 0, 0] == 10.0  # y=0, t=0, x=0
    assert foo.temp[0, 0, 1] == 20.0  # y=0, t=0, x=1
    assert foo.temp[1, 1, 0] == 30.0  # y=1, t=1, x=0
    assert foo.temp[1, 1, 1] == 40.0  # y=1, t=1, x=1


def test_sparse_dict_converter_empty_dict():
    """Test behavior with empty dict when field is optional."""

    @xattree
    class Foo:
        t: int = dim()
        x: int = dim()
        temp: NDArray[np.float64] | None = array(
            dims=("t", "x"),
            converter=dict_to_array_converter,
            default=None,
        )

    foo = Foo(t=2, x=2, temp={})
    assert foo.temp is None
