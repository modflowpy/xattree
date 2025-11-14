"""
Tests documenting runtime type-checking behavior for child relationships.

The library implements STRICT type checking:
1. Single child fields: Reject any type that doesn't match the type hint
2. Child collections: Raise TypeError for items that don't match the type hint
3. Union types: Accept any union member, reject types not in the union
4. Non-xattree objects: Reject with AttributeError

All type violations raise clear TypeError exceptions.
"""

import pytest

from xattree import field, xattree


@xattree
class ChildA:
    """A child type for testing."""

    value: int = field(default=1)


@xattree
class ChildB:
    """A different child type for testing."""

    value: int = field(default=2)


@xattree
class ChildC:
    """Yet another child type for testing."""

    value: int = field(default=3)


class NotAChild:
    """A regular class, not decorated with xattree."""

    pass


# ============================================================================
# Tests documenting actual behavior for single child fields
# ============================================================================


def test_single_child_rejects_wrong_xattree_type():
    """Single child fields reject xattree types that don't match the type hint."""

    @xattree
    class Parent:
        child: ChildA = field()

    # Cannot pass ChildB where ChildA is expected
    child_b = ChildB(value=99)

    with pytest.raises(
        TypeError, match="Cannot initialize field 'child'.*ChildB.*expected.*ChildA"
    ):
        Parent(child=child_b)


def test_single_child_rejects_non_xattree():
    """Single child fields reject non-xattree objects."""

    @xattree
    class Parent:
        child: ChildA = field()

    not_a_child = NotAChild()

    # Should raise TypeError for type mismatch
    with pytest.raises(
        TypeError, match="Cannot initialize field 'child'.*NotAChild.*expected.*ChildA"
    ):
        Parent(child=not_a_child)


def test_single_child_rejects_primitives():
    """Single child fields reject primitive types."""

    @xattree
    class Parent:
        child: ChildA = field()

    # Should raise TypeError for type mismatch
    with pytest.raises(TypeError, match="Cannot initialize field 'child'.*str.*expected.*ChildA"):
        Parent(child="not a child")


# ============================================================================
# Tests documenting actual behavior for child list collections
# ============================================================================


def test_child_list_rejects_wrong_types_on_init():
    """Child lists reject items that don't match the type hint during initialization."""

    @xattree
    class Parent:
        child_list: list[ChildA] = field()

    child_a = ChildA(value=1)
    child_b = ChildB(value=2)  # Wrong type

    # Initialize with mixed types raises TypeError
    with pytest.raises(
        TypeError,
        match="Cannot initialize field 'child_list'.*ChildB.*at index 1.*expected.*ChildA",
    ):
        Parent(child_list=[child_a, child_b])


def test_child_list_rejects_wrong_types_on_append():
    """Child lists reject appended items that don't match the type hint."""

    @xattree
    class Parent:
        child_list: list[ChildA] = field()

    parent = Parent(child_list=[ChildA(value=1)])
    child_b = ChildB(value=99)  # Wrong type

    # Try to append wrong type raises TypeError
    with pytest.raises(TypeError, match="Cannot add ChildB to child_list.*expected.*ChildA"):
        parent.child_list.append(child_b)


def test_child_list_rejects_wrong_types_on_setitem():
    """Child lists reject items set via indexing that don't match the type hint."""

    @xattree
    class Parent:
        child_list: list[ChildA] = field()

    parent = Parent(child_list=[ChildA(value=1)])
    child_b = ChildB(value=99)  # Wrong type

    # Try to set wrong type via index raises TypeError
    with pytest.raises(TypeError, match="Cannot add ChildB to child_list.*expected.*ChildA"):
        parent.child_list[0] = child_b


def test_child_list_rejects_wrong_types_on_replace():
    """Child lists reject wrong types when replacing the entire list."""

    @xattree
    class Parent:
        child_list: list[ChildA] = field()

    parent = Parent(child_list=[ChildA(value=1)])
    child_b = ChildB(value=99)  # Wrong type

    # Try to replace with wrong type raises TypeError
    with pytest.raises(
        TypeError, match="Cannot add ChildB to field 'child_list'.*at index 0.*expected.*ChildA"
    ):
        parent.child_list = [child_b]


def test_child_list_rejects_non_xattree():
    """Child lists reject non-xattree objects."""

    @xattree
    class Parent:
        child_list: list[ChildA] = field()

    parent = Parent(child_list=[ChildA(value=1)])
    not_a_child = NotAChild()

    # Should raise TypeError for type mismatch
    with pytest.raises(TypeError, match="Cannot add NotAChild to child_list.*expected.*ChildA"):
        parent.child_list.append(not_a_child)


def test_child_list_rejects_primitives():
    """Child lists reject primitive types."""

    @xattree
    class Parent:
        child_list: list[ChildA] = field()

    parent = Parent(child_list=[ChildA(value=1)])

    # Should raise TypeError for type mismatch
    with pytest.raises(TypeError, match="Cannot add str to child_list.*expected.*ChildA"):
        parent.child_list.append("not a child")


# ============================================================================
# Tests documenting actual behavior for child dict collections
# ============================================================================


def test_child_dict_rejects_wrong_types_on_init():
    """Child dicts reject items that don't match the type hint during initialization."""

    @xattree
    class Parent:
        child_dict: dict[str, ChildA] = field()

    child_a = ChildA(value=1)
    child_b = ChildB(value=2)  # Wrong type

    # Initialize with mixed types raises TypeError
    with pytest.raises(
        TypeError,
        match="Cannot initialize field 'child_dict'.*ChildB.*at key 'b'.*expected.*ChildA",
    ):
        Parent(child_dict={"a": child_a, "b": child_b})


def test_child_dict_rejects_wrong_types_on_setitem():
    """Child dicts reject items set that don't match the type hint."""

    @xattree
    class Parent:
        child_dict: dict[str, ChildA] = field()

    parent = Parent(child_dict={"a": ChildA(value=1)})
    child_b = ChildB(value=99)  # Wrong type

    # Try to set wrong type raises TypeError
    with pytest.raises(TypeError, match="Cannot add ChildB to child dict.*expected.*ChildA"):
        parent.child_dict["b"] = child_b


def test_child_dict_rejects_wrong_types_on_replace():
    """Child dicts reject wrong types when replacing the entire dict."""

    @xattree
    class Parent:
        child_dict: dict[str, ChildA] = field()

    parent = Parent(child_dict={"a": ChildA(value=1)})
    child_b = ChildB(value=99)  # Wrong type

    # Try to replace with wrong type raises TypeError
    with pytest.raises(
        TypeError, match="Cannot add ChildB to field 'child_dict'.*expected.*ChildA"
    ):
        parent.child_dict = {"b": child_b}


def test_child_dict_rejects_non_xattree():
    """Child dicts reject non-xattree objects."""

    @xattree
    class Parent:
        child_dict: dict[str, ChildA] = field()

    parent = Parent(child_dict={"a": ChildA(value=1)})
    not_a_child = NotAChild()

    # Should raise TypeError for type mismatch
    with pytest.raises(TypeError, match="Cannot add NotAChild to child dict.*expected.*ChildA"):
        parent.child_dict["b"] = not_a_child


def test_child_dict_rejects_primitives():
    """Child dicts reject primitive types."""

    @xattree
    class Parent:
        child_dict: dict[str, ChildA] = field()

    parent = Parent(child_dict={"a": ChildA(value=1)})

    # Should raise TypeError for type mismatch
    with pytest.raises(TypeError, match="Cannot add str to child dict.*expected.*ChildA"):
        parent.child_dict["b"] = "not a child"


# ============================================================================
# Tests documenting actual behavior for union types
# ============================================================================


def test_union_accepts_all_union_members():
    """Union types accept any of the union members."""

    @xattree
    class Parent:
        child_list: list[ChildA | ChildB] = field()

    child_a = ChildA(value=1)
    child_b = ChildB(value=2)

    # Both union members are accepted
    parent = Parent(child_list=[child_a, child_b])
    assert len(parent.child_list) == 2
    assert isinstance(parent.child_list[0], ChildA)
    assert isinstance(parent.child_list[1], ChildB)


def test_union_rejects_types_not_in_union():
    """Union types reject types not in the union."""

    @xattree
    class Parent:
        child_list: list[ChildA | ChildB] = field()

    child_a = ChildA(value=1)
    child_c = ChildC(value=3)  # Not in union

    # ChildC raises TypeError since it's not in the union
    with pytest.raises(
        TypeError, match="Cannot initialize field 'child_list'.*ChildC.*at index 1.*expected"
    ):
        Parent(child_list=[child_a, child_c])
