from typing import Optional

import pytest
from attrs import define

from xattree import field, xattree


@xattree
class Child:
    pass


def test_child_default_factory():
    @xattree
    class Parent:
        child: Child = field()

    parent = Parent()
    assert parent.child is not None


def test_child_default_none():
    @xattree
    class Parent:
        child: Optional[Child] = field(default=None)

    parent = Parent()
    assert parent.child is None


def test_child_access():
    @xattree
    class Parent:
        child: Child = field()

    child = Child()
    parent = Parent(child=child)
    assert parent.child is child
    assert parent.data["child"] is child.data


def test_child_replace():
    @xattree
    class Child:
        i: int = field(default=0)

    @xattree
    class Parent:
        child: Child = field()

    child = Child()
    parent = Parent(child=child)
    parent.child = Child(i=1)
    assert parent.data["child"].i == 1
    assert parent.data["child"].equals(parent.child.data)


def test_child_list_default_factory():
    @xattree
    class Parent:
        child_list: list[Child] = field()

    parent = Parent()
    assert parent.child_list == []


def test_child_list_default_none():
    with pytest.raises(ValueError, match=r".*default may not be None.*"):

        @xattree
        class Parent:
            child_list: list[Child] = field(default=None)


def test_child_list_access():
    @xattree
    class Parent:
        child_list: list[Child] = field()

    children = Child(), Child()
    parent = Parent(child_list=children)
    assert parent.child_list[0] is children[0]
    assert parent.child_list[1] is children[1]
    assert parent.child_list[0] == children[0]
    assert parent.child_list[1] == children[1]


def test_child_list_append():
    @xattree
    class Parent:
        child_list: list[Child] = field()

    children = Child(), Child()
    parent = Parent(child_list=children)
    parent.child_list.append(Child(name="child_list2"))
    assert len(parent.child_list) == 3
    assert parent.data["child_list2"].equals(parent.child_list[2].data)


def test_child_list_setitem():
    @xattree
    class Child:
        i: int = field(default=0)

    @xattree
    class Parent:
        child_list: list[Child] = field()

    children = Child(), Child()
    parent = Parent(child_list=children)
    parent.child_list[0] = Child(i=1)
    assert parent.data["child_list0"].i == 1
    assert parent.data["child_list0"].equals(parent.child_list[0].data)


def test_child_list_replace():
    @xattree
    class Child:
        i: int = field(default=0)

    @xattree
    class Parent:
        child_list: list[Child] = field()

    children = Child(), Child()
    parent = Parent(child_list=children)
    parent.child_list = [Child(i=1)]
    assert parent.data["child_list0"].i == 1
    assert parent.data["child_list0"].equals(parent.child_list[0].data)
    assert len(parent.child_list) == 1


def test_child_dict_default_factory():
    @xattree
    class Parent:
        child_dict: dict[str, Child] = field()

    parent = Parent()
    assert parent.child_dict == {}


def test_child_dict_default_none():
    with pytest.raises(ValueError, match=r".*default may not be None.*"):

        @xattree
        class Parent:
            child_dict: dict[str, Child] = field(default=None)


def test_child_dict_access():
    @xattree
    class Parent:
        child_dict: dict[str, Child] = field()

    children = Child(), Child()
    parent = Parent(child_dict={"child0": children[0], "child1": children[1]})
    assert parent.child_dict["child0"] is children[0]
    assert parent.child_dict["child1"] is children[1]
    assert parent.child_dict["child0"] == children[0]
    assert parent.child_dict["child1"] == children[1]


def test_child_dict_setitem():
    @xattree
    class Parent:
        child_dict: dict[str, Child] = field()

    children = Child(), Child()
    parent = Parent(child_dict={"child0": children[0], "child1": children[1]})
    parent.child_dict["child2"] = Child()
    assert len(parent.child_dict) == 3
    assert parent.data["child2"].equals(parent.child_dict["child2"].data)


def test_child_dict_replace():
    @xattree
    class Child:
        i: int = field(default=0)

    @xattree
    class Parent:
        child_dict: dict[str, Child] = field()

    children = Child(), Child()
    parent = Parent(child_dict={"child0": children[0], "child1": children[1]})
    parent.child_dict = {"child2": Child(i=1)}
    assert parent.data["child2"].i == 1
    assert parent.data["child2"].equals(parent.child_dict["child2"].data)
    assert len(parent.child_dict) == 1


def test_multiple_child_fields_same_type():
    @xattree
    class Parent:
        children_a: dict[str, Child] = field()
        children_b: dict[str, Child] = field()

    parent = Parent()
    assert parent.children_a == {}
    assert parent.children_b == {}


def test_multiple_child_fields_different_types():
    @define(slots=False)
    class ChildA:
        pass

    @define(slots=False)
    class ChildB:
        pass

    @xattree
    class Parent:
        children_a: dict[str, ChildA] = field()
        children_b: dict[str, ChildB] = field()

    parent = Parent()
    assert parent.children_a == {}
    assert parent.children_b == {}


def test_field_may_not_be_named_children():
    with pytest.raises(Exception):

        @xattree
        class Parent:
            children: list[Child] = field()


class ChildNotAttrs:
    pass


def test_list_of_not_attrs():
    """
    If a field is a list whose value type is not `attrs`, it should be
    registered not as a child but as an arbitrary attribute.
    """

    @xattree
    class Parent:
        child_list: list[ChildNotAttrs] = field()

    children = [Child()]
    parent = Parent(child_list=children)
    assert parent.child_list is children
    assert parent.data.attrs["child_list"] is children
    assert not any(parent.data.children)


def test_dict_of_not_attrs():
    """
    If a field is a dictionary whose value type is not `attrs`, it should
    be registered not as a child but as an arbitrary attribute.
    """

    @xattree
    class Parent:
        child_dict: dict[str, ChildNotAttrs] = field()

    children = {"0": Child()}
    parent = Parent(child_dict=children)
    assert parent.child_dict is children
    assert parent.data.attrs["child_dict"] is children
    assert not any(parent.data.children)


def test_reserved_field_names():
    class Parent:
        pass

    with pytest.raises(ValueError, match=r".*reserved.*"):

        @xattree
        class Grandparent:
            parent: Parent = field()


def test_nested_children():
    @xattree
    class Parent:
        child: Child = field()

    @xattree
    class Grandparent:
        parent_: Parent = field()

    child = Child()
    parent = Parent(child=child)
    grandparent = Grandparent(parent_=parent)

    assert grandparent.parent_.child is child
    assert grandparent.data["parent_"]["child"] is child.data


def test_nested_child_lists():
    @xattree
    class Parent:
        child_list: list[Child] = field()

    @xattree
    class Grandparent:
        child_list: list[Parent] = field()

    children = [Child(), Child()]
    parent = Parent(child_list=children)
    grandparent = Grandparent(child_list=[parent])

    assert grandparent.child_list[0] is parent
    assert grandparent.child_list[0].child_list[0] is children[0]


def test_nested_child_dicts():
    @xattree
    class Parent:
        child_dict: dict[str, Child] = field()

    @xattree
    class Grandparent:
        child_dict: dict[str, Parent] = field()

    children = {"0": Child(), "1": Child()}
    parent = Parent(child_dict=children)
    grandparent = Grandparent(child_dict={"0": parent})

    assert grandparent.child_dict["0"] is parent
    assert grandparent.child_dict["0"].child_dict["0"] is children["0"]
    assert grandparent.data["0"] is parent.data
    assert grandparent.data["0"]["0"] is parent.data["0"]
    assert children["0"].data.attrs["host"] == children["0"]


def test_optional_child_with_value():
    """Optional child field should work when provided a valid child instance."""

    @xattree
    class OptionalChild:
        i: int = field(default=0)

    @xattree
    class Parent:
        child: Optional[OptionalChild] = field(default=None)

    child = OptionalChild(i=42)
    parent = Parent(child=child)
    assert parent.child is child
    assert parent.child.i == 42
    assert parent.data["child"] is child.data


def test_optional_child_with_none():
    """Optional child field should work when set to None."""

    @xattree
    class OptionalChild:
        pass

    @xattree
    class Parent:
        child: Optional[OptionalChild] = field(default=None)

    parent = Parent()
    assert parent.child is None

    parent = Parent(child=None)
    assert parent.child is None


def test_optional_child_union_syntax():
    """Optional child field using X | None syntax should work."""

    @xattree
    class OptionalChild:
        i: int = field(default=0)

    @xattree
    class Parent:
        child: OptionalChild | None = field(default=None)

    parent = Parent()
    assert parent.child is None

    child = OptionalChild(i=99)
    parent = Parent(child=child)
    assert parent.child is child
    assert parent.child.i == 99


def test_optional_child_replace_with_none():
    """Should be able to replace optional child with None."""

    @xattree
    class OptionalChild:
        i: int = field(default=0)

    @xattree
    class Parent:
        child: Optional[OptionalChild] = field(default=None)

    child = OptionalChild(i=5)
    parent = Parent(child=child)
    assert parent.child is child

    parent.child = None
    assert parent.child is None


def test_optional_child_replace_none_with_value():
    """Should be able to replace None with a child instance."""

    @xattree
    class OptionalChild:
        i: int = field(default=0)

    @xattree
    class Parent:
        child: Optional[OptionalChild] = field(default=None)

    parent = Parent()
    assert parent.child is None

    child = OptionalChild(i=10)
    parent.child = child
    assert parent.child is child
    assert parent.data["child"] is child.data


def test_optional_child_in_list():
    """List of optional child should not be allowed (lists handle optionality)."""

    @xattree
    class OptionalChild:
        pass

    # This should work - list elements can be the child type
    @xattree
    class Parent:
        children_: list[OptionalChild] = field()

    parent = Parent()
    assert parent.children_ == []


def test_multiple_optional_children():
    """Multiple optional child fields should all work correctly."""

    @xattree
    class ChildA:
        a: int = field(default=1)

    @xattree
    class ChildB:
        b: int = field(default=2)

    @xattree
    class Parent:
        child_a: Optional[ChildA] = field(default=None)
        child_b: Optional[ChildB] = field(default=None)

    parent = Parent()
    assert parent.child_a is None
    assert parent.child_b is None

    child_a = ChildA(a=10)
    child_b = ChildB(b=20)
    parent = Parent(child_a=child_a, child_b=child_b)
    assert parent.child_a is child_a
    assert parent.child_b is child_b
    assert parent.child_a.a == 10
    assert parent.child_b.b == 20


def test_nested_optional_children():
    """Optional child containing another optional child should work."""

    @xattree
    class GrandChild:
        i: int = field(default=0)

    @xattree
    class MiddleChild:
        grandchild: Optional[GrandChild] = field(default=None)

    @xattree
    class Parent:
        child: Optional[MiddleChild] = field(default=None)

    # All None
    parent = Parent()
    assert parent.child is None

    # Parent has child, but child's grandchild is None
    middle = MiddleChild()
    parent = Parent(child=middle)
    assert parent.child is middle
    assert parent.child.grandchild is None

    # Full hierarchy
    grandchild = GrandChild(i=100)
    middle = MiddleChild(grandchild=grandchild)
    parent = Parent(child=middle)
    assert parent.child.grandchild is grandchild
    assert parent.child.grandchild.i == 100
