# Tutorial

`xattree` maps your objects onto `xarray`'s powerful data model. Given the former, it gives it back, and everything acts (more or less) the same, but behind it all is a (network of) `xarray.DataTree` node(s). You also get

- a `DataTree`-like API for connecting parents to children and vice versa
- dimension and coordinate inheritance
- hierarchical addressing

..and the other goodies offered by `xarray` more generally.

## Decorate classes

Decorate any `attrs`-based class with `@xattree` instead of [`@define`](https://www.attrs.org/en/stable/examples.html#basics). The former wraps the latter.

Stop here and you won't notice a difference. But there is acrimony under your feet. Rays of sun and hostile glances filter through bare branches.

**Note**: `xattree` must use `slots=False` for embarrassing reasons. This library is probably not for you if you need tens of thousands of instances.

## Decorate fields

Replace `attrs.field()` with `xattree.dim()`, `coord()`, and `array()`, and `field()` as appropriate. 

The rules:

- use `coord()` if your variable is itself a dimension coordinate array
- use `dim()` if your variable is a scalar indicating the size of a dimension
- use `array()` for array variables, ideally specifying their shape by reference to `dims` defined in the same or another `xattree`-decorated class
- use `field()` for arbitrary attributes or child components &mdash; the former go into `DataTree.attrs`, the latter are described in more detail below

The clamor recedes to the rustle of many claws grappling for place. A hierarchy forms, seemingly of its own accord. Soon there is peace.

**Note**: `xattree` tries to follow the `xarray` [data model](https://docs.xarray.dev/en/latest/user-guide/terminology.html) and its conventions. Notable among these is the fact that a dimension may not live separately from a coordinate or data array. Thus a solitary `dim()` indicates a dimension coordinate, and you get an eponymous coordinate array in the `DataTree`.

**Note**: `xattree` assumes you intend to handle `attrs.field`s separately and ignores them. Use `attrs.field()` for attributes you don't want `xattree` to know about.

### Children

At import time, `xattree` walks your domain to discover its structure. Where it discovers a `field()` whose type is either another `xattree` node or an `Optional`, `Mapping` or `Iterable` of such, it inspects it recursively.

When a field is another `xattree`-decorated class, an `Optional` of such, it simply becomes a child node in the data tree.

When it's a `Mapping` or `Iterable` of some `xattree`-decorated class, we get a bit fancy. `xattree` will "flatten" these before attaching them. Basically, use a `Mapping` or `Iterable` if you don't want to name your children up front (i.e. when you define your object model) but at their moment of birth, which is perhaps understandable. Using an `Iterable` is effectively to declare that you don't want to have to name them, which, while deplorable among our kind, is standard feline conduct, therefore `xattree` grudgingly accepts but insists on naming anonymous children behind your back, appending an auto-incrementing integer to the name of their field.

**Note**: While `xattree` will raise an error at runtime if a user-specified or auto-generated name collides with another fields, it's best to name fields such that collisions are impossible.
