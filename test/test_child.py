from typing import Optional

import pytest
from attrs import field

from xattree import xattree


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
    assert parent.data["child_list_0"] is children[0].data
    assert parent.data["child_list_1"] is children[1].data


@pytest.mark.xfail(reason="TODO implement list mutations")
def test_child_list_mutate():
    @xattree
    class Parent:
        child_list: list[Child] = field()

    children = Child(), Child()
    parent = Parent(child_list=children)
    parent.child_list.append(Child())
    assert len(parent.child_list) == 3
    assert parent.data["child_list_2"] is parent.child_list[2].data


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


@pytest.mark.xfail(reason="TODO implement dict mutations")
def test_child_dict_mutate():
    @xattree
    class Parent:
        child_dict: dict[str, Child] = field()

    children = Child(), Child()
    parent = Parent(child_dict={"child0": children[0], "child1": children[1]})
    parent.child_dict["child2"] = Child()
    assert len(parent.child_dict) == 3
    assert parent.data["child2"] is parent.child_dict["child2"].data
