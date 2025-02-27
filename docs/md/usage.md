# Usage

Hand us `attrs` classes with some extra metadata, we hand them back.

They look and act the same. But each one now has an `xarray.DataTree` in `.data` (by default).

## Class decorator

Decorate a class with `@xattree` instead of [`@define`](https://www.attrs.org/en/stable/examples.html#basics).

## Field decorators

If this is where you stop, you won't notice a difference.

But there is growing acrimony under your feet. Rays of sun filter through the bare branches of your tree.

To rectify this, swap out `field()` for `dim()`, `coord()`, and `array()` as appropriate. `xattree` follows the `xarray` [data model](https://docs.xarray.dev/en/latest/user-guide/terminology.html) and its conventions &mdash; e.g., a scalar `dim()` is interpreted as a dimension coordinate, and adds an eponymous coordinate array

### Dimensions

TODO

### Coordinates

TODO

### Arrays

TODO

### Children

TODO

#### Collections

Child collections give you more flexibility in naming your children. Dictionaries and lists are supported. `xattree` will "flatten" dictionaries and lists of child types before attaching them to the `DataTree`.

Use a dictionary of child types if you want to name them not up front but at their moment of birth, which is understandable. 

Using a list is effectively to declare that you don't want to have to name them, which, while deplorable among our kind, is standard feline conduct, therefore `xattree` accepts anonymous children. But it will name them behind your back, appending an auto-incrementing integer to the name of the list field.

While `xattree` will raise an error at runtime if a user-specified or auto-generated name collides with another fields, it's best to name fields such that collisions are impossible.
