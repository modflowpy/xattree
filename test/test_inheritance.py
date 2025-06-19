import numpy as np
import pytest
from numpy.typing import NDArray

from xattree import array, dim, get_xatspec, xattree


def test_subclass():
    @xattree
    class Base:
        pass

    @xattree
    class Child(Base):
        n: int = dim(default=3)
        a: NDArray[np.floating] = array(dims=("n",), default=1.0)

    spec = get_xatspec(Child).flat
    assert len(spec) == 2

    child = Child(n=3)
    assert child.n == 3
    assert isinstance(child.n, int)
    assert np.array_equal(child.a, np.array([1.0, 1.0, 1.0]))


def test_subclass_inherits_fields():
    @xattree
    class Base:
        n: int = dim(default=3)
        a: NDArray[np.floating] = array(dims=("n",), default=1.0)

    @xattree
    class Child(Base):
        pass

    spec = get_xatspec(Child).flat
    assert len(spec) == 2

    child = Child(n=3)
    assert child.n == 3
    assert isinstance(child.n, int)
    assert np.array_equal(child.a, np.array([1.0, 1.0, 1.0]))


@pytest.mark.xfail(reason="TODO")
def test_subclass_must_be_decorated():
    @xattree
    class Base:
        pass

    with pytest.raises(TypeError):

        class Child(Base):
            pass
