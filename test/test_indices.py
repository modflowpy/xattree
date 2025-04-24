import numpy as np
from numpy.typing import NDArray
from xarray.core.indexes import Index, PandasIndex
from xarray.core.indexing import merge_sel_results

from xattree import ROOT, Indices, array, dim, xattree


@xattree(index=lambda ds: Indices.alias_dim(ds, "n", "i"))
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


class GridIndex(Index):
    def __init__(self, indices):
        dims = [idx.dim for idx in indices.values()]
        assert len(dims) == 2
        assert dims[0] != dims[1]
        self._indices = indices

    @classmethod
    def from_variables(cls, variables):
        assert len(variables) == 2
        return {k: PandasIndex.from_variables({k: v}) for k, v in variables.items()}

    def create_variables(self, variables=None):
        idx_vars = {}
        for index in self._indices.values():
            idx_vars.update(index.create_variables(variables))
        return idx_vars

    def sel(self, labels):
        results = []
        for k, index in self._indices.items():
            if k in labels:
                results.append(index.sel({k: labels[k]}))
        return merge_sel_results(results)


@xattree(
    index=lambda ds: GridIndex(
        {"i": Indices.alias_dim(ds, "rows", "i"), "j": Indices.alias_dim(ds, "cols", "j")}
    )
)
class Grid:
    rows: int = dim(scope=ROOT, default=3, coord=False)
    cols: int = dim(scope=ROOT, default=3, coord=False)
    nodes: int = dim(scope=ROOT, init=False)
    a: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))
    aa: NDArray[np.float64] = array(default=0.0, dims=("nodes",))

    def __attrs_post_init__(self):
        self.nodes = self.rows * self.cols


def test_grid_index():
    grid = Grid()
    assert grid.rows == 3
    assert grid.cols == 3
    assert grid.nodes == 9
    assert grid.data.i.shape == (3,)
    assert grid.data.j.shape == (3,)
    assert "i" in grid.data.coords
    assert "j" in grid.data.coords
    assert "rows" not in grid.data.coords
    assert "cols" not in grid.data.coords
    assert grid.data.a.shape == (3, 3)
    assert grid.data.aa.shape == (9,)
