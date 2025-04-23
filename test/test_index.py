import numpy as np
import pandas as pd
from numpy.typing import NDArray
from xarray.core.indexes import PandasIndex

from xattree import array, dim, xattree


def make_index(ds, src_name, tgt_name):
    return PandasIndex(pd.RangeIndex(ds.sizes[src_name], name=tgt_name), dim=src_name)


@xattree(index=lambda ds: make_index(ds, "n", "nn"))
class Foo:
    n: int = dim(default=3, coord=False)
    a: NDArray[np.float64] = array(default=0.0, dims=("n",))


def test_index():
    foo = Foo()
    assert foo.n == 3
    assert foo.a.shape == (3,)
    assert foo.data.nn.shape == (3,)
    assert "nn" in foo.data.coords
    assert "n" not in foo.data.coords
