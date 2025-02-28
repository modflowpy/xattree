# Tutorial

The basic idea is: hand us `attrs` classes with some extra metadata, we hand them back.

They look and act the same. Each now has an `xarray.DataTree` in `.data` (by default).

## Decorate class(es)

Decorate any `attrs`-based class with `@xattree` instead of [`@define`](https://www.attrs.org/en/stable/examples.html#basics).

## Decorate fields

If this is where you stop, you won't notice a difference. But there is acrimony under your feet. Rays of sun filter through bare branches. You receive hostile glances.

Simply swap out `field()` for `dim()`, `coord()`, and `array()` as appropriate. The clamor recedes to the rustle of many claws grappling for place. A hierarchy forms, seemingly of its own accord. Soon there is peace.

The rules:

- use `dim()` if your variable indicates the size of a dimension
- use `coord()` if your variable is a dimension coordinate
- use `array()` for data variables, ideally specifying `dims`
- use `field()` for arbitrary attributes or child components

`xattree` tries to follow the `xarray` [data model](https://docs.xarray.dev/en/latest/user-guide/terminology.html) and its conventions. Notable among these is the fact that a dimension may not live separately from a coordinate or data array. Thus `xattree` expands `dim()` into a dimension coordinate, and you get an eponymous coordinate array.

### Children

At import time, `xattree` walks your domain to discover its nested structure. Where it discovers a `field()` whose type is either another `xattree` node or an `Optional`, `Mapping` or `Iterable` of such, it inspects it recursively.

When a field is another `xattree`-decorated class, an `Optional` of such, it becomes a child node in the data tree.

A `Mapping` or `Iterable` of children is also supported. `xattree` will "flatten" these before attaching them as children to the given class' `DataTree` node. Use a `Mapping` or `Iterable` if you don't want to name your children up front (i.e. when you define your object model) but at their moment of birth, which is perhaps understandable. Using an `Iterable` is effectively to declare that you don't want to have to name them, which, while deplorable among our kind, is standard feline conduct. Therefore `xattree` accepts but will name anonymous children behind your back, appending an auto-incrementing integer to the name of the field.

While `xattree` will raise an error at runtime if a user-specified or auto-generated name collides with another fields, it's best to name fields such that collisions are impossible.
