# # Basics

# A Python class walks into a bar.

import numpy as np
from numpy.typing import NDArray
from xattree import xattree, has, dim, array

class Foo:
    n: int = dim(default=10)
    a: NDArray[np.float64] = array(default=0., dims=("n",))

def bar(cls):
    return cls if has(cls) else xattree(cls)

FooBar = bar(Foo)

# **Note**: don't use the function form, just decorate your classes &mdash; unless you have good reason, like a joke to make

# A short while later it emerges, acting more or less the same, but carrying itself differently &mdash; more *together*, somehow.

fubar = FooBar()
fubar.a

# You sense deception. Maybe this is not your typical dive, but one of those places young people go for over-priced caffeine.

# A few more, these with strange hats and ominous attributes, stride in. You begin to fret.

from xattree import ROOT, field

@xattree
class Grid:
    x: int = dim(default=3, scope=ROOT)
    y: int = dim(default=3, scope=ROOT)

@xattree
class Arrs:
    a: NDArray[np.float64] = array(default=0.0, dims=("x", "y"))

@xattree
class Root:
    grid: Grid = field()
    arrs: Arrs = field()

grid = Grid()
root = Root(grid=grid)
arrs = Arrs(parent=root)

# Soon something struts out, doing them a perfect imitation. Be not fooled.

print(root.data)
