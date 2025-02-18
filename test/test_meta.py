from xattree import xats, xattree


@xattree
class Foo:
    pass


class Bar:
    pass


def test_xat():
    assert xats(Foo)
    assert not xats(Bar)
