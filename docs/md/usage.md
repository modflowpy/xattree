# Usage

Hand us `attrs` classes with some extra metadata, we hand them back.

They look and act the same. But each one now has an `xarray.DataTree` in `.data` (by default).

## Class decorator

Decorate a class with `@xattree` instead of [`@define`](https://www.attrs.org/en/stable/examples.html#basics).

## Field decorators

If this is where you stop, you won't notice a difference.

But there is growing acrimony under your feet. Rays of sun filter through the bare branches of your tree.

Simply swap out `field()` for `dim()`, `coord()`, and `array()` as appropriate. These engender a certain context-sensitivity &mdash; useful for keeping oneself and one's attributes in alignment with one's superiors. The clamor at your feet dies down as a hierarchy forms, seemingly of its own accord. Soon there is peace.

### Dimensions

`xattree` tries to follow the `xarray` [data model](https://docs.xarray.dev/en/latest/user-guide/terminology.html) and its conventions. Notable among these is the fact that a dimension may not live separately from a coordinate or data array. Thus `xattree` interprets a class with a scalar `dim()` as a dimension coordinate, and you get an eponymous coordinate array.

### Coordinates

TODO

### Arrays

TODO

### Children

At import time, `xattree` walks your domain to discover its nested structure.

Where it discovers a `field()` whose type is either another `xattree` node or an `Optional`, `Mapping` or `Iterable` of such, it recursively inspects it.

`xattree` treats children differently depending on their cardinality. A single child is handled straightforwardly.

A `Mapping` or `Iterable` of children is also supported. `xattree` will "flatten" these before attaching them to the `DataTree` &mdash; i.e., items will not be attached under a new node, but directly to the current node, and distinguished by name. But standard attribute access still works.

Use a `Mapping` or `Iterable` if you don't want to name your children up front (i.e. when you define your object model) but at their moment of birth, which is perhaps understandable. 

Using an `Iterable` is effectively to declare that you don't want to have to name them, which, while deplorable among our kind, is standard feline conduct. Therefore `xattree` accepts but will name anonymous children behind your back, appending an auto-incrementing integer to the name of the field.

While `xattree` will raise an error at runtime if a user-specified or auto-generated name collides with another fields, it's best to name fields such that collisions are impossible.
