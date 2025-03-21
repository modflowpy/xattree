# xattree

[![CI](https://github.com/modflowpy/xattree/actions/workflows/ci.yml/badge.svg)](https://github.com/modflowpy/xattree/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/xattree/badge/?version=latest)](https://xattree.readthedocs.io/en/latest/?badge=latest)
[![GitHub contributors](https://img.shields.io/github/contributors/modflowpy/xattree)](https://img.shields.io/github/contributors/modflowpy/xattree)

`attrs` + `xarray.DataTree` = `xattree`

"exa-tree", or "cat tree" if you like.

```python
import numpy as np
from numpy.typing import NDArray
from xattree import xattree, dim, array, field, ROOT 

@xattree
class Grid:
    rows: int = dim(scope=ROOT, default=3)
    cols: int = dim(scope=ROOT, default=3)

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
<xarray.DataTree 'root'>
Group: /
│   Dimensions:  (rows: 3, cols: 3)
│   Coordinates:
│     * rows      (rows) int64 24B 0 1 2
│     * cols      (cols) int64 24B 0 1 2
├── Group: /grid
│       Attributes:
│           rows:     3
│           cols:     3
└── Group: /arrs
        Dimensions:  (rows: 3, cols: 3)
        Data variables:
            arr      (rows, cols) float64 72B 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
```
