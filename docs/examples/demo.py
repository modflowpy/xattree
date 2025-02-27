# # Demo

# A Python class walks into a bar.

import numpy as np
from numpy.typing import NDArray
from xattree import xattree, has_xats, dim, array

class Foo:
    n: int = dim(default=10)
    arr: NDArray[np.float64] = array(default=0., dims=("n",))

def bar(cls):
    return cls if has_xats(cls) else xattree(cls)

FooBar = bar(Foo)

# A short while later it emerges, acting more or less the same, but carrying itself differently. It's got itself *together*, somehow &mdash; now in a different dimension, so to speak.

fubar = FooBar()
fubar.arr

# You sense deception. Maybe this is not your typical dive, but one of those places young people go to mainline over-priced caffeine, strive in one another's company, and negotiate which well-established words to overload in the interest of fun, fundraising, or faux-bohemian effect. You watch more closely.

# Some more classes with strange hats and ominous-looking attributes wander in. You begin to fret.

from attrs import field
from xattree import ROOT 

@xattree
class Grid:
    rows: int = dim(name="row", scope=ROOT, default=3)
    cols: int = dim(name="col", scope=ROOT, default=3)

@xattree
class Arrs:
    arr: NDArray[np.float64] = array(default=0.0, dims=("row", "col"))

@xattree
class Root:
    grid: Grid = field()
    arrs: Arrs = field()

grid = Grid()
root = Root(grid=grid)
arrs = Arrs(parent=root)

# Soon a `xarray.DataTree` struts out, doing them a perfect imitation.

root.data

# **Note**: don't use the function form, just decorate your classes &mdash; unless you have good reason, like a joke to make.