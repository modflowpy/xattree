import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from xarray.core.indexes import PandasIndex

from xattree import array, dim, xattree


def alias(dataset: xr.Dataset, dim_name: str, idx_name: str) -> PandasIndex:
    return PandasIndex(pd.RangeIndex(dataset.sizes[dim_name], name=idx_name), dim=dim_name)


@xattree(index=lambda ds: alias(ds, "n", "i"))
class Foo:
    n: int = dim(default=3, coord=False)
    a: NDArray[np.float64] = array(default=0.0, dims=("n",))


def test_simple_index():
    foo = Foo()
    assert foo.n == 3
    assert foo.a.shape == (3,)
    assert foo.data.a.shape == (3,)
    assert foo.data.i.shape == (3,)
    assert "i" in foo.data.coords
    assert "n" not in foo.data.coords
