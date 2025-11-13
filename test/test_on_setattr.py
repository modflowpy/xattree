"""Test on_setattr hook functionality."""

import pytest
from attrs import setters

from xattree import array, dim, field, xattree


def test_field_on_setattr_hook_called():
    """Test that on_setattr hooks are called when setting field attributes."""
    call_count = {"count": 0}

    def track_calls(inst, attrib, new_value):
        call_count["count"] += 1
        return new_value

    @xattree
    class WithHook:
        value: str = field(default="initial", on_setattr=track_calls)

    obj = WithHook()
    assert call_count["count"] == 0  # Hook shouldn't be called during init

    # This should trigger the hook
    obj.value = "modified"
    assert call_count["count"] == 1
    assert obj.value == "modified"

    # Setting again should trigger it again
    obj.value = "modified_again"
    assert call_count["count"] == 2
    assert obj.value == "modified_again"


def test_field_on_setattr_with_attrs_setters():
    """Test on_setattr with attrs built-in setters like setters.frozen."""

    @xattree
    class FrozenField:
        frozen_value: str = field(default="immutable", on_setattr=setters.frozen)
        mutable_value: str = field(default="mutable")

    obj = FrozenField()

    # Should be able to set mutable field
    obj.mutable_value = "changed"
    assert obj.mutable_value == "changed"

    # Should raise error when trying to set frozen field
    with pytest.raises(AttributeError):
        obj.frozen_value = "attempt_to_change"


def test_array_on_setattr_hook_called():
    """Test that on_setattr hooks are called when setting array fields."""
    call_count = {"count": 0}

    def track_array_calls(inst, attrib, new_value):
        call_count["count"] += 1
        return new_value

    @xattree
    class WithArrayHook:
        n: int = dim()
        arr: list[float] = array(dims=("n",), on_setattr=track_array_calls)

    obj = WithArrayHook(n=3, arr=[1.0, 2.0, 3.0])
    assert call_count["count"] == 0  # Hook shouldn't be called during init

    # This should trigger the hook
    obj.arr = [4.0, 5.0, 6.0]
    assert call_count["count"] == 1

    # Setting again should trigger it again
    obj.arr = [7.0, 8.0, 9.0]
    assert call_count["count"] == 2


def test_on_setattr_receives_correct_arguments():
    """Test that on_setattr hook receives the correct instance, attribute, and value."""
    captured = {"inst": None, "attrib": None, "value": None}

    def capture_args(inst, attrib, new_value):
        captured["inst"] = inst
        captured["attrib"] = attrib
        captured["value"] = new_value
        return new_value

    @xattree
    class CaptureArgs:
        value: str = field(default="initial", on_setattr=capture_args)

    obj = CaptureArgs()
    obj.value = "new_value"

    assert captured["inst"] is obj
    assert captured["attrib"].name == "value"
    assert captured["value"] == "new_value"


def test_on_setattr_can_transform_value():
    """Test that on_setattr hook can transform the value before setting."""

    def uppercase_value(inst, attrib, new_value):
        return new_value.upper() if isinstance(new_value, str) else new_value

    @xattree
    class TransformValue:
        text: str = field(default="hello", on_setattr=uppercase_value)

    obj = TransformValue()
    assert obj.text == "hello"  # Default is not transformed

    obj.text = "world"
    assert obj.text == "WORLD"  # Hook transforms the value

    obj.text = "test"
    assert obj.text == "TEST"


def test_on_setattr_multiple_fields():
    """Test that multiple fields can have different on_setattr hooks."""
    field1_calls = {"count": 0}
    field2_calls = {"count": 0}

    def track_field1(inst, attrib, new_value):
        field1_calls["count"] += 1
        return new_value

    def track_field2(inst, attrib, new_value):
        field2_calls["count"] += 1
        return new_value

    @xattree
    class MultipleHooks:
        field1: str = field(default="a", on_setattr=track_field1)
        field2: str = field(default="b", on_setattr=track_field2)

    obj = MultipleHooks()

    obj.field1 = "modified"
    assert field1_calls["count"] == 1
    assert field2_calls["count"] == 0

    obj.field2 = "modified"
    assert field1_calls["count"] == 1
    assert field2_calls["count"] == 1


def test_on_setattr_with_validation():
    """Test that on_setattr works together with validation logic."""

    def validate_positive(inst, attrib, new_value):
        if new_value < 0:
            raise ValueError(f"{attrib.name} must be positive")
        return new_value

    @xattree
    class ValidatedField:
        value: int = field(default=10, on_setattr=validate_positive)

    obj = ValidatedField()

    obj.value = 20
    assert obj.value == 20

    with pytest.raises(ValueError, match="value must be positive"):
        obj.value = -5


def test_on_setattr_not_called_during_init():
    """Test that on_setattr hooks are NOT called during initialization."""
    call_count = {"count": 0}

    def track_calls(inst, attrib, new_value):
        call_count["count"] += 1
        return new_value

    @xattree
    class InitTest:
        value: str = field(default="default", on_setattr=track_calls)

    # Create with default value
    obj1 = InitTest()
    assert call_count["count"] == 0

    # Create with explicit value
    obj2 = InitTest(value="explicit")  # noqa: F841
    assert call_count["count"] == 0

    # Only when setting after init should it be called
    obj1.value = "modified"
    assert call_count["count"] == 1


def test_on_setattr_pipe():
    """Test using attrs setters.pipe to chain multiple setters."""
    call_log = []

    def log_setter(name):
        def setter(inst, attrib, new_value):
            call_log.append(name)
            return new_value

        return setter

    @xattree
    class PipedSetters:
        value: str = field(
            default="initial",
            on_setattr=setters.pipe(
                log_setter("first"),
                log_setter("second"),
                log_setter("third"),
            ),
        )

    obj = PipedSetters()
    call_log.clear()

    obj.value = "new"
    assert call_log == ["first", "second", "third"]
    assert obj.value == "new"
