# # Indexes

# The mechanism powering `xarray` [label-based lookups](https://docs.xarray.dev/en/stable/user-guide/indexing.html) is called an "index". The most flexible way to extend indexing and selection is to [create a custom index](https://docs.xarray.dev/en/v2025.01.0/internals/how-to-create-custom-index.html). For simpler cases, e.g. "aliasing" dimension coordinates, there are more straightforward approaches.
# 
# Consider an `attrs` class describing a 2D structured grid , with integer fields for the size of each dimension and an array field containing some data variable living on grid cells. This is a canonical case for `xattree` and the `dim()` and `array()` decorators.

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from xattree import array, dim, xattree

@xattree
class Grid:
    rows: int = dim(default=3)
    cols: int = dim(default=3)
    arr: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))

grid = Grid()
grid.data

# But the field names aren't quite right as coordinate names &mdash; while it's natural for `rows` and `cols` to become [dimension coordinates](https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Dimension-coordinate) and for dimensions to be named such, we expect `i` and `j` as coordinate names.
# 
# We might think first to try `Dataset.rename()`:

grid.data.dataset.rename({"rows": "i", "cols": "j"})

# But this renames not only the coordinates but also the dimensions. Ideally, we want dimensions `rows`/`cols`, coordinates `i`/`j`.

# `xattree` provides a simple way to achieve this: just pass a new name to the `coord` parameter of the `dim()` decorator. This will create a new coordinate variable with the given name, but leave the dimension name unchanged. As expected for a dimension coordinate, the new coordinate variable will have a `PandasIndex`` attached.

@xattree
class Grid:
    rows: int = dim(default=3, coord="i")
    cols: int = dim(default=3, coord="j")
    arr: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))

grid = Grid()
grid.data

# A more general approach is to create a custom index.

# As a first step, we can set `coord=False` on the `dim()` call, which will prevent `xattree` from creating a coordinate variable for a dimension field.

@xattree
class Grid:
    rows: int = dim(default=3, coord=False)
    cols: int = dim(default=3, coord=False)
    arr: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))

# Now the `rows` and `cols` fields are not coordinates, but the dataset now has no coordinate variables.

grid = Grid()
grid.data

# To support `i` and `j` labels, we can create a custom index class and register it with `xattree`.
#
# A custom index is a class subclassing `Index`. The `Index` interface is a set of methods that allow you to create, manipulate, and query the index. 
#
# For our purposes, an index which simply aliases the `rows` and `cols` fields to `i` and `j`, respectively, will suffice.
#
# First create an aliasing function, which creates for a dataset dimension with the given name a `PandasIndex` with a new name.

from xarray.core.indexes import Index, PandasIndex

def alias(dataset: xr.Dataset, old_name: str, new_name: str) -> PandasIndex:
    """Alias a dimension coordinate to a coordinate with a different name."""
    try:
        size = dataset.sizes[old_name]
    except KeyError:
        size = dataset.attrs[old_name]
    return PandasIndex(pd.RangeIndex(size, name=new_name), dim=old_name)

# Now create a [meta-index](https://docs.xarray.dev/en/stable/internals/how-to-create-custom-index.html#meta-indexes), with which we can combine two "aliased" 1D indexes into a 2D index.

from xarray.core.indexing import merge_sel_results

class MetaIndex(Index):
    def __init__(self, indices):
        self._indices = indices

    @classmethod
    def from_variables(cls, variables):
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

# Define a function to build an instance of the index from a dataset.

def meta_index(dataset: xr.Dataset) -> MetaIndex:
    return MetaIndex(
        {
            "i": alias(dataset, "rows", "i"),
            "j": alias(dataset, "cols", "j"),
        }
    )

# Finally, register it with the `xattree` decorator. The `index` parameter is a callable that takes a dataset and returns an index.

@xattree(index=meta_index)
class Grid:
    rows: int = dim(default=3, coord=False)
    cols: int = dim(default=3, coord=False)
    arr: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))

grid = Grid()
grid.data

# Try out the new label-based indexing.

grid.data.arr.sel(i=0)

# Make sure the index is working as expected.

assert grid.rows == 3
assert grid.cols == 3
assert grid.data.i.shape == (3,)
assert grid.data.j.shape == (3,)
assert "i" in grid.data.coords
assert "j" in grid.data.coords
assert "rows" not in grid.data.coords
assert "cols" not in grid.data.coords
assert grid.data.arr.shape == (3, 3)

# Like dimensions and coordinates, an index can be "lifted" to a context above the current class with the `xattree` decorator's `index_scope` parameter.

from xattree import ROOT, field

@xattree(index=meta_index, index_scope=ROOT)
class Grid:
    rows: int = dim(default=3, coord=False, scope=ROOT)
    cols: int = dim(default=3, coord=False, scope=ROOT)

@xattree
class Arrs:
    arr: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))

@xattree
class Root:
    grid: Grid = field()
    arrs: Arrs = field()

grid = Grid()
root = Root(grid=grid)
arrs = Arrs(parent=root)

root.data

# Indexing still works as expected.

assert arrs.arr.sel(i=0).shape == (3,)

# Derived dimensions are supported too.

def meta_index(dataset: xr.Dataset) -> MetaIndex:
    return MetaIndex(
        {
            "i": alias(dataset, "rows", "i"),
            "j": alias(dataset, "cols", "j"),
            "n": alias(dataset, "nodes", "n"),
        }
    )

@xattree(index=meta_index, index_scope=ROOT)
class Grid:
    rows: int = dim(default=3, coord=False, scope=ROOT)
    cols: int = dim(default=3, coord=False, scope=ROOT)
    nodes: int = dim(init=False, coord=False, scope=ROOT)

    def __attrs_post_init__(self):
        self.nodes = self.rows * self.cols

@xattree
class Arrs:
    arr_a: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))
    arr_b: NDArray[np.float64] = array(default=0.0, dims=("nodes",))

@xattree
class Root:
    grid: Grid = field()
    arrs: Arrs = field()

grid = Grid()
root = Root(grid=grid)
arrs = Arrs(parent=root)

root.data