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
    assert parent.data["child_list0"] is children[0].data
    assert parent.data["child_list1"] is children[1].data


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
    assert parent.data["child0"] is children[0].data
    assert parent.data["child1"] is children[1].data


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
    assert grandparent.data["0"]["0"] is children["0"].data
    assert children["0"].data._host == children["0"]  # impl detail
