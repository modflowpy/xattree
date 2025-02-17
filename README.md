# xattree

`attrs` + `xarray.DataTree` = `xattree`

"exa-tree", or "cat tree" if you like.

```python
import numpy as np
from numpy.typing import NDArray
from attrs import field, Factory
from xattree import xattree, dim, array, child

@xattree
class Grid:
    rows: int = dim(coord="j", scope="root", default=3)
    cols: int = dim(coord="i", scope="root", default=3)

@xattree
class Arrs:
    arr: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))

@xattree
class Root:
    grid = child(Grid)
    arrs = child(Arrs)

grid = Grid()
root = Root(grid=grid)
arrs = Arrs(root)
root.data
<xarray.DataTree 'root'>
Group: /
│   Dimensions:  (rows: 3, cols: 3)
│   Coordinates:
│       j        (rows) int64 24B 0 1 2
│       i        (cols) int64 24B 0 1 2
│   Dimensions without coordinates: rows, cols
├── Group: /grid
│       Dimensions:  (rows: 3, cols: 3)
│       Coordinates:
│           j        (rows) int64 24B 0 1 2
│           i        (cols) int64 24B 0 1 2
│       Dimensions without coordinates: rows, cols
│       Attributes:
│           rows:     3
│           cols:     3
└── Group: /arrs
        Dimensions:  (rows: 3, cols: 3)
        Coordinates:
            j        (rows) int64 24B 0 1 2
            i        (cols) int64 24B 0 1 2
        Dimensions without coordinates: rows, cols
        Data variables:
            arr      (rows, cols) float64 72B 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
```