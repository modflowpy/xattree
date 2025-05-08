import numpy as np
import pytest
from numpy.typing import NDArray

from xattree import _get_xatspec, array, dim, has_xats, xattree


def test_child_redecorated():
    @xattree
    class Base:
        pass

    @xattree
    class Child(Base):
        n: int = dim(default=3)
        a: NDArray[np.floating] = array(dims=("n",), default=1.0)

    spec = _get_xatspec(Child)
    assert len(spec.flat) == 2

    child = Child(n=3)
    assert child.n == 3
    assert isinstance(child.n, int)
    assert np.array_equal(child.a, np.array([1.0, 1.0, 1.0]))


@pytest.mark.xfail(reason="TODO")
def test_child_inherited_decorator():
    @xattree
    class Base:
        pass

    class Child(Base):
        n: int = dim(default=3)
        a: NDArray[np.floating] = array(dims=("n",), default=1.0)

    spec = _get_xatspec(Child)
    assert len(spec.flat) == 2

    child = Child(n=3)
    assert child.n == 3
    assert isinstance(child.n, int)
    assert np.array_equal(child.a, np.array([1.0, 1.0, 1.0]))


@pytest.mark.xfail(reason="TODO")
def test_child_inherited_decorator_and_fields():
    @xattree
    class Base:
        n: int = dim(default=3)
        a: NDArray[np.floating] = array(dims=("n",), default=1.0)

    class Child(Base):
        pass

    spec = _get_xatspec(Child)
    assert len(spec.flat) == 2

    child = Child(n=3)
    assert child.n == 3
    assert isinstance(child.n, int)
    assert np.array_equal(child.a, np.array([1.0, 1.0, 1.0]))


@pytest.mark.xfail(reason="TODO")
def test_child_disabled_inheritance():
    @xattree(inherit=False)
    class Base:
        n: int = dim(default=3)
        a: NDArray[np.floating] = array(dims=("n",), default=1.0)

    class Child(Base):
        pass

    assert not has_xats(Child)
