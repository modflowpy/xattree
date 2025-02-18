# What?

A Python class walks into a bar.

```python
import numpy as np
from numpy.typing import NDArray
from attrs import field, Factory
from xattree import xattree, dim, array, child

class Foo:
    num: int = dim(default=10)
    baz: NDArray[float] = array(dims=("num",))

def bar(cls):
    return xattree(Foo)

FooBar = bar(Foo)
```

A short while later it emerges, acting the same, but carrying itself differently. It's got itself *together*, somehow. In a different dimension, so to speak.

```python
bar.baz
```

You sense deception. Maybe this is not your typical dive, but one of those places young people go to mainline over-priced caffeine, strive in one another's company, and overload well-established words for faux-bohemian effect. You watch more closely.

Some more classes with ominous-looking attributes wander in.

```python
class Qux:
   pass

class Quux:
   pass
```

Soon a `xarray.DataTree` struts out, wearing their clothes, doing them a perfect imitation. A small crowd, in tow, cheers. You begin to fret.

Don't. It's just a cat tree.

And don't use the function form &mdash; just decorate your classes &mdsah; unless you have good reason, like a joke to make.