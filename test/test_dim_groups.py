import numpy as np
import pytest
from numpy.typing import NDArray

from xattree import _get_xatspec, array, dim, xattree


def test_dim_group_parameter():
    """Test that the group parameter works correctly for dim fields."""

    @xattree
    class TestClass:
        x: int = dim(group="space", default=3)
        y: int = dim(group="time", default=4)
        z: int = dim(group=None, default=5)  # Explicitly None
        w: int = dim(default=6)  # No group parameter

    # Test that the class can be instantiated
    instance = TestClass()

    # Test that the values are correct
    assert instance.x == 3
    assert instance.y == 4
    assert instance.z == 5
    assert instance.w == 6

    # Test that the specifications have the correct group attributes
    spec = _get_xatspec(TestClass)

    assert spec.dims["x"].group == "space"
    assert spec.dims["y"].group == "time"
    assert spec.dims["z"].group is None
    assert spec.dims["w"].group is None


def test_array_dim_groups():
    """Test that array fields have correct dim_groups computed."""

    @xattree
    class TestClass:
        time: int = dim(group="time", default=3)
        lat: int = dim(group="space", default=4)
        lon: int = dim(group="space", default=5)
        depth: int = dim(default=6)  # No group

        # Array with grouped dimensions
        temp: NDArray[np.float64] = array(dims=("time", "lat", "lon", "depth"), default=0.0)

        # Array with only one group
        spatial_data: NDArray[np.float64] = array(dims=("lat", "lon"), default=1.0)

        # Array with ungrouped dimensions
        simple_data: NDArray[np.float64] = array(dims=("depth",), default=2.0)

    spec = _get_xatspec(TestClass)

    # Check temp array dim_groups
    temp_array = spec.arrays["temp"]
    assert temp_array.dims == ("time", "lat", "lon", "depth")
    assert temp_array.dim_groups == ("time", "space", "space", None)

    # Check spatial_data array dim_groups
    spatial_array = spec.arrays["spatial_data"]
    assert spatial_array.dims == ("lat", "lon")
    assert spatial_array.dim_groups == ("space", "space")

    # Check simple_data array dim_groups
    simple_array = spec.arrays["simple_data"]
    assert simple_array.dims == ("depth",)
    assert simple_array.dim_groups == (None,)


def test_array_dim_groups_validation():
    """Test that non-disjoint group ordering in array dims raises an error."""

    with pytest.raises(ValueError, match="not disjointly ordered by group"):

        @xattree
        class InvalidClass:
            time: int = dim(group="time", default=3)
            lat: int = dim(group="space", default=4)
            lon: int = dim(group="space", default=5)

            # This should fail because we have space, then time, then space again
            # which makes the space group non-contiguous
            a: NDArray[np.float64] = array(dims=("lat", "time", "lon"), default=0.0)


def test_array_dim_groups_class_order_irrelevant():
    """Test that class definition order doesn't matter, only array dims order."""

    @xattree
    class TestClass:
        # Define dimensions in mixed order in the class
        lat: int = dim(group="space", default=4)
        time: int = dim(group="time", default=2)
        lon: int = dim(group="space", default=5)

        # Array dims are properly ordered by group even though class defs aren't
        a: NDArray[np.float64] = array(dims=("time", "lat", "lon"), default=0.0)

    spec = _get_xatspec(TestClass)
    data_array = spec.arrays["a"]

    # Should work fine because the array dims are properly grouped
    assert data_array.dim_groups == ("time", "space", "space")


def test_array_dim_groups_multiple_groups():
    """Test arrays with multiple groups in correct order."""

    @xattree
    class TestClass:
        time: int = dim(group="time", default=2)
        lat: int = dim(group="space", default=4)
        lon: int = dim(group="space", default=5)
        depth1: int = dim(group="vertical", default=6)
        depth2: int = dim(group="vertical", default=7)
        species: int = dim(default=8)  # No group

        a: NDArray[np.float64] = array(
            dims=("time", "lat", "lon", "depth1", "depth2", "species"), default=0.0
        )

    spec = _get_xatspec(TestClass)
    data_array = spec.arrays["a"]

    assert data_array.dim_groups == ("time", "space", "space", "vertical", "vertical", None)
