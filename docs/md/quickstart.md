# Quickstart

A Python class walks into a bar.

```python
import numpy as np
from numpy.typing import NDArray
from attrs import field, Factory
from xattree import xattree, xat, dim, array, child

class Foo:
    num: int = dim(coord="n", default=10)
    arr: NDArray[np.float64] = array(default=0., dims=("num",))

def bar(cls):
    return Foo if xat(Foo) else xattree(Foo)

FooBar = bar(Foo)
```

A short while later it emerges, acting the same, but carrying itself differently. It's got itself *together*, somehow. In a different dimension, so to speak.

```python
>>> fubar = FooBar()
fubar.arr
<xarray.DataArray 'arr' (num: 10)> Size: 80B
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
Coordinates:
    n        (num) int64 80B 0 1 2 3 4 5 6 7 8 9
Dimensions without coordinates: num
```

You sense deception. Maybe this is not your typical dive, but one of those places young people go to mainline over-priced caffeine, strive in one another's company, and negotiate which well-established word to overload next in the interest of fun, fundraising, or faux-bohemian effect. You watch more closely.

Some more classes with strange hats and ominous-looking attributes wander in. You begin to fret.

```python
@xattree
class Grid:
    rows: int = dim(coord="j", scope="root", default=3)
    cols: int = dim(coord="i", scope="root", default=3)

@xattree
class Arrs:
    arr: NDArray[np.float64] = array(default=0.0, dims=("rows", "cols"))

@xattree
class Root:
    grid: Grid = child(Grid)
    arrs: Arrs = child(Arrs)

grid = Grid()
root = Root(grid=grid)
arrs = Arrs(parent=root)
```

Soon a `xarray.DataTree` struts out, doing them a perfect imitation.

```python
>>> root.data
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

Plumbing nested objects with shared dimensions is like herding cats: give them a good tree and they'll sort themselves out.

**Note**: don't use the function form &mdash; just decorate your classes &mdash; unless you have good reason, like a joke to make.