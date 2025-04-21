# Tutorial

`xattree` maps your objects onto `xarray`'s data model. Given the former, it gives them back. They act the same, but behind it all is a (network of) `xarray.DataTree` node(s). You get

- a `Dataset` for each instance living in `.data` by default
- dimension and coordinate inheritance
- hierarchical addressing

..and the goodies offered by `xarray` more generally.

## Class decorator

Decorate an `attrs`-based class with `@xattree` instead of [`@define`](https://www.attrs.org/en/stable/examples.html#basics).

Stop here and you won't notice a difference. But there is acrimony under your feet. Rays of sun and hostile glances filter through your tree's bare branches.

## Field decorators

Replace `attrs.field()` with `xattree.dim()`, `coord()`, and `array()`, and `field()` as appropriate. Claws grapple for place. A hierarchy forms, seemingly of its own accord. Soon there is peace.

The rules:

- use `coord()` if your variable is a dimension coordinate array
- use `dim()` if your variable is an integer indicating the size of a dimension
- use `array()` for array variables, ideally specifying their shape by reference to `dims` defined in the same or another `xattree`-decorated class
- use `field()` for arbitrary attributes or child components &mdash; the former go into `DataTree.attrs`, the latter are described in more detail below
- Use `attrs.field()` for attributes you want `xattree` to ignore &mdash; these will not be moved from your object's `__dict__` to the tree

When you initialize an instance with an array field, `xattree` will try to look up the array's shape by its dimension names, if you have specified any. The size of each dimension may be a field in another component somewhere above the current class in your object hierarchy. If you provide an array value, the shape will be checked. If you don't provide a value at init time, and the array has a default (scalar) value, it will be expanded with `np.full` to the proper shape.

`xattree` adds several "hidden" fields to your objects, called `name`, `dims`, `parent`, `children`, and `strict`. These names are reserved and may not be reused by your fields.

Some of these become "hidden" named `__init__` parameters &mdash; though they are accepted, they will not appear in intellisense.

- `name`: `__init__` parameter. Use it to set the name of the instance's `DataTree` node.

- `dims`: `__init__` parameter. Use it to explicitly provide dimension sizes, overriding or replacing dimension lookup.

- `parent`: `__init__` parameter. Use it to set the instance's parent.

- `children`: **Not** an `__init__` parameter, just a read-only field. Returns the instance's children.

- `strict`: `__init__` parameter. Defaults true. Set this to false to turn dimension/alignment checking off, i.e. to allow creating arrays whose dimensions can't be found (thus size can't be verified).

**Note**: type hints are required for all fields.

**Note**: `xattree` wraps `attrs.define()` with `slots=False` for embarrassing reasons. This library is probably not for you if you need lots of instances of your classes.

**Note**: `xattree` tries to follow the `xarray` [data model](https://docs.xarray.dev/en/latest/user-guide/terminology.html) and its conventions. Notable among these is the fact that a dimension may not live separately from a coordinate or data array. Thus a solitary `dim()` indicates a dimension coordinate, and you get an eponymous coordinate array in the `DataTree`.

### Children

At import time, `xattree` walks your domain to discover its structure. Where it discovers a `field()` whose type is either another `xattree` node or an `Optional`, `Mapping` or `Iterable` of such, it inspects it recursively.

When a field is another `xattree`-decorated class, an `Optional` of such, it simply becomes a child node in the data tree. When it's a `Mapping` or `Iterable` of some `xattree`-decorated class, `xattree` will "flatten" these before attaching them to the data tree &mdash; i.e., attach each element by name as a child instead of attaching the collection itself as a child.

Use a `Mapping` or `Iterable` if you don't want to name your children up front (i.e. when you define your object model) but at their moment of birth, which is perhaps understandable. Using an `Iterable` is effectively to declare that you don't want to have to name them, which, while deplorable among our kind, is standard feline conduct, therefore `xattree` grudgingly accepts but insists on naming anonymous children behind your back, appending an auto-incrementing integer to the name of their field.

**Note**: While `xattree` will raise an error at runtime if a user-specified or auto-generated name collides with another fields, it's best to name fields such that collisions are impossible.

**Note**: You may not reassign a child to a different parent if it already has one.

### Conversion and validation

Like `attrs`, `xattree` supports automatic conversion of field values using the `converter` parameter, and field validation using the `validator` parameter. This can be useful for mapping values from a format convenient for user input to a more canonical type, e.g. converting "sparse" list input into an array.

**Note**: array conversion and validation runs *after* the [`attrs` initialization procedure](https://www.attrs.org/en/stable/init.html#order-of-execution) is complete. All other conversions/validations are piped through the `attrs` mechanisms. Provided you use an `attrs.Converter` with `takes_self=True`, this gives your array conversion functions access to the instance `__dict__` and everything sent to it through `__init__` method arguments, including explicit dimensions and/or parent components whose dimensions the given component may inherit.

## Rearranging the furniture

By default, `xattree` names the datatree attribute `data`. To name it something else, use `xattree(where="...")`.

**Note**: some names are reserved, namely the core `xattree`-managed fields (`name`, `parent`, `children`, `dims`, and `strict`).

**Note**: unlike typical appointments of wood and fabric, wherever you put the tree, that's where it stays. If you try to move it (at runtime), things will break. If you regret your choice, tough luck &mdash; find a new apartment (i.e. kill the program and set `where` to something else). 