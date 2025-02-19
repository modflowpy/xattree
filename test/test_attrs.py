"""Test `attrs` integration."""

from typing import ClassVar

from xattree import xattree


def test_init_hooks():
    """Make sure pre- and post-init hooks are still called."""

    @xattree
    class Hooks:
        pre: ClassVar[bool] = False
        post: ClassVar[bool] = False

        def __attrs_pre_init__(self):
            Hooks.pre = True

        def __attrs_post_init__(self):
            Hooks.post = True

    assert not Hooks.pre
    assert not Hooks.post
    Hooks()
    assert Hooks.pre
    assert Hooks.post
