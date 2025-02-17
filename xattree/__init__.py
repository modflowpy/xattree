import builtins
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any, Optional, get_origin

import numpy as np
from attr import Attribute, fields_dict
from attrs import has
from beartype.claw import beartype_this_package
from beartype.vale import Is, IsAttr, IsInstance
from numpy.typing import ArrayLike, NDArray
from xarray import DataArray, Dataset, DataTree

beartype_this_package()


class DimsNotFound(KeyError):
    """Raised if an array variable specifies dimensions that can't be found."""

    pass


class ExpandFailed(ValueError):
    """
    Raised if a scalar default is provided for an array variable
    specifying no dimensions. The scalar can't be expanded to an
    array without a known shape.
    """

    pass


_Numeric = int | float | np.int64 | np.float64
"""A numeric value."""

_Scalar = bool | _Numeric | str | Path
"""A scalar value."""

_Array = list | np.ndarray
"""An array value."""

_HasAttrs = Annotated[object, Is[lambda obj: has(type(obj))]]
"""Runtime-applied type hint for `attrs` based class instances."""

_HasTree = Annotated[object, IsAttr["data", IsInstance[DataTree]]]
"""Runtime-applied type hint for objects with a `DataTree` in `.data`."""

_Xattree = Annotated[
    object,
    (Is[lambda obj: has(type(obj))] & IsAttr["data", IsInstance[DataTree]]),
]
"""
An `attrs`-based class with a `DataTree` in `.data`.
"""


def get(
    tree: DataTree, key: str, default: Optional[Any] = None
) -> Optional[Any]:
    """Get a scalar or array value from a `DataTree`."""
    value = tree.get(key, None)
    match value:
        case DataTree():
            return value
        case DataArray():
            return value.item() if value.shape == () else value
        case None:
            pass
        case _:
            raise ValueError(f"Unexpected value type: {type(value)}")
    value = tree.dims.get(key, None)
    if value is not None:
        return value
    value = tree.attrs.get(key, None)
    if value is not None:
        return value
    return default


def chexpand(value: ArrayLike, shape: tuple[int]) -> Optional[NDArray]:
    """If `value` is iterable, check its shape. If scalar, expand to shape."""
    value = np.array(value)
    if value.shape == ():
        return np.full(shape, value.item())
    if value.shape != shape:
        raise ValueError(
            f"Shape mismatch, got {value.shape}, expected {shape}"
        )
    return value


def split(spec: Mapping[str, Attribute]) -> dict[str, dict]:
    """
    Split a class' `attrs` specification into scalars (including dimensions,
    coordinates, and unbound scalars), arrays (including coordinates), and
    children (subcomponents).
    """
    spec_ = {
        "dimensions": {},
        "coordinates": {},
        "scalars": {},
        "arrays": {},
        "children": {},
    }

    for var in spec.values():
        bind = var.metadata.get("bind", False)
        dims = var.metadata.get("dims", None)
        dim = var.metadata.get("dim", None)
        coord = var.metadata.get("coord", None)
        default = var.default

        if dim and coord:
            raise ValueError(
                f"Variable '{var.name}' cannot have "
                f"both 'dim' and 'coord' metadata."
            )

        match var.type:
            # array
            case t if t and issubclass(get_origin(t) or object, _Array):
                spec_["arrays"][var.name] = var
                if coord:
                    dim_name = (
                        coord.get("dim", var.name)
                        if isinstance(coord, dict)
                        else var.name
                    )
                    spec_["dimensions"][dim_name] = default
                    spec_["coordinates"][dim_name] = {
                        "name": var.name,
                        "scope": coord.get("scope", None),
                    }
            # scalar
            case t if t and issubclass(t, _Scalar):
                assert dims is None
                spec_["scalars"][var.name] = var
                if dim:
                    spec_["dimensions"][var.name] = default
                    if not isinstance(dim, dict):
                        continue
                    spec_["coordinates"][var.name] = {
                        "name": dim.get("coord", var.name),
                        "scope": dim.get("scope", None),
                    }
            # child
            case t if t:
                assert bind
                assert dims is None
                assert has(t)
                spec_["children"][var.name] = var
            case _:
                raise ValueError(f"Variable has no type: {var.name}")

    return spec_


def bind_tree(
    self: _Xattree,
    parent: _Xattree = None,
    children: Optional[Mapping[str, _Xattree]] = None,
):
    """Bind a cat tree to its parent and children."""
    name = self.data.name
    children = children or {}

    # bind parent
    if parent:
        if name in parent.data:
            parent.data.update({name: self.data})
            parent.data.self = parent
        else:
            parent.data = parent.data.assign({name: self.data})
            parent.data.self = parent

        self.data = parent.data[self.data.name]
        self.parent = parent

    # bind children
    for n, child in children.items():
        self.data.update({n: child.data})
        child.data = self.data[n]
        self.data[n].self = child
        self.data[n].self.parent = self

    # give the data tree a reference to the instance
    # so it can be the class hierarchy's "backbone",
    # i.e. so that an instance can be accessed from
    # another instance's data tree in `getattribute`.
    # TODO: think thru the consequences here. how to
    # avoid memory leaks?
    self.data.self = self


def init_tree(
    self: _HasAttrs,
    name: Optional[str] = None,
    parent: Optional[_HasTree] = None,
    children: Optional[Mapping[str, _HasTree]] = None,
    dimensions: Optional[Mapping[str, int]] = None,
    coordinates: Optional[Mapping[str, ArrayLike]] = None,
    strict: bool = True,
):
    """
    Initialize a cat tree.

    Notes
    -----
    This method must run after the default `__init__()`.

    The tree is built from the class' `attrs` fields, i.e.
    spirited from the instance's `__dict__` into the tree,
    which is attached to the instance as `self.data`. The
    `__dict__` is empty after this method runs except for
    the data tree. Field access is proxied to the tree.

    The class cannot use slots for this to work.
    """

    cls = type(self)
    cls_name = cls.__name__.lower()
    spec = split(fields_dict(cls))
    scalars = {}
    arrays = {}
    coordinates = coordinates or {}
    dimensions = dimensions or {}
    children = children or {}

    def _yield_children():
        for var in spec["children"].values():
            bind = var.metadata.get("bind", False)
            if bind:
                val = self.__dict__.pop(var.name, None)
                yield (var.name, val)

    children = dict(list(_yield_children())) | children

    def _yield_scalars():
        for var in spec["scalars"].values():
            val = self.__dict__.pop(var.name, var.default)
            yield (var.name, val)

    scalars = dict(list(_yield_scalars()))

    def _resolve_array(
        attr: Attribute,
        value: ArrayLike,
        strict: bool = False,
        **kwargs,
    ) -> tuple[Optional[NDArray], Optional[dict[str, NDArray]]]:
        dims = attr.metadata.get("dims", None)
        if dims is None and (value is None or isinstance(value, _Scalar)):
            raise ExpandFailed(
                f"Class '{cls_name}' array "
                f"'{attr.name}' can't expand, no dims."
            )
        elif value is None:
            value = attr.default
        elif dims is None:
            return value
        shape = [kwargs.pop(dim, dim) for dim in dims]
        unresolved = [dim for dim in shape if not isinstance(dim, int)]
        if any(unresolved):
            if strict:
                raise DimsNotFound(
                    f"Class 'cls_name' array "
                    f"'{attr.name}' failed dim resolution: "
                    f"{', '.join(unresolved)}"
                )
            return None
        array = chexpand(value, shape)
        return array

    def _yield_arrays():
        inherited_dims = dict(parent.data.dims) if parent else {}
        for var in spec["arrays"].values():
            dims = var.metadata.get("dims", None)
            coord = var.metadata.get("coord", None)
            if coord:
                continue
            array = _resolve_array(
                var,
                value=self.__dict__.pop(var.name, var.default),
                strict=strict,
                **(inherited_dims | dimensions),
            )
            if array is None:
                continue
            yield (var.name, (dims, array) if dims else array)

    arrays = dict(list(_yield_arrays()))

    def _yield_coords():
        # coords are either explicitly provided
        # arrays, inferred/expanded from local
        # dims, or inherited from parent tree
        if parent:
            for coord_name, coord in parent.data.coords.items():
                yield (coord_name, (coord.dims, coord.data))
        for var in spec["arrays"].values():
            coord = var.metadata.get("coord", None)
            if not coord:
                continue
            dim_name = coord.get("dim", var.name)
            array = _resolve_array(
                var,
                value=self.__dict__.pop(var.name, var.default),
                strict=strict,
            )
            if array is None:
                continue
            yield (var.name, (dim_name, array))
        for scalar_name, scalar in scalars.items():
            dim_name = scalar_name
            dim_size = scalar
            if dim_name not in spec["dimensions"]:
                continue
            coord = spec["coordinates"][scalar_name]
            match type(scalar):
                case builtins.int | np.int64:
                    step = coord.get("step", 1)
                    start = 0
                case builtins.float | np.float64:
                    step = coord.get("step", 1.0)
                    start = 0.0
                case _:
                    raise ValueError("Dimensions/coordinates must be numeric.")
            yield (
                coord.get("name", scalar_name),
                (dim_name, np.arange(start, dim_size, step)),
            )

    coordinates = dict(list(_yield_coords())) | coordinates

    self.data = DataTree(
        Dataset(
            data_vars=arrays,
            coords=coordinates,
            attrs={
                n: v
                for n, v in scalars.items()  # if n not in dimensions
            },
        ),
        name=name or cls.__name__.lower(),
        children={n: c.data for n, c in children.items()},
    )

    bind_tree(self, parent=parent, children=children)


def getattribute(self: _Xattree, name: str) -> Any:
    """
    Proxy `attrs` attribute access, returning values from
    an `xarray.DataTree` in `self.data`.

    Notes
    -----
    Override `__getattr__` with this in classes fulfilling
    the `_Xattree` contract.
    """
    if name == "data":
        raise AttributeError

    cls = type(self)
    spec = fields_dict(cls)
    tree = self.data
    var = spec.get(name, None)
    if var:
        value = get(tree, name, None)
        if isinstance(value, DataTree):
            return value.self
        if value is not None:
            return value

    if name == "parent":
        return None

    raise AttributeError


def pop_children(
    self: _HasAttrs, **kwargs
) -> tuple[dict[str, _HasAttrs], dict[str, Any]]:
    children = {}
    kwargs_copy = kwargs.copy()
    spec = fields_dict(type(self))
    for name, value in kwargs_copy.items():
        match = spec.get(name, None)
        if match and has(match.type) and match.type is type(value):
            children[name] = kwargs.pop(name)
    return children, kwargs


def yield_coords(
    scope: str, **kwargs
) -> Iterator[tuple[str, tuple[str, Any]]]:
    for value in kwargs.values():
        cls = type(value)
        if not has(cls):
            continue
        spec = fields_dict(cls)
        for n, var in spec.items():
            coord = var.metadata.get("coord", None)
            if coord:
                if scope == coord.get("scope", None):
                    yield coord.get("dim", n), (n, value.data.coords[n].data)
            dim = var.metadata.get("dim", None)
            if dim:
                if scope == dim.get("scope", None):
                    coord_name = dim.get("coord", n)
                    yield coord_name, (n, value.data.coords[coord_name].data)


def xattree(maybe_cls: Optional[type[_HasAttrs]] = None) -> type[_Xattree]:
    """
    Make an `attrs`-based class a (node in a) cat tree.

    Notes
    -----
    For this to work, the class cannot use slots.
    """

    def validate(spec):
        reserved = ["name", "dims", "strict"]
        for name in reserved:
            if name in spec:
                raise ValueError(
                    f"A field may not be named '{name}', "
                    f"reserved names are: {', '.join(reserved)}"
                )

    def wrap(cls):
        spec = fields_dict(cls)
        validate(spec)
        init_self = cls.__init__
        cls_name = cls.__name__.lower()

        def init(self, *args, **kwargs):
            name = kwargs.pop("name", cls_name)
            parent = args[0] if args and any(args) else None
            children, kwargs = pop_children(self, **kwargs)
            dimensions = kwargs.pop("dims", {})
            coordinates = dict(list(yield_coords(scope=cls_name, **children)))
            strict = kwargs.pop("strict", False)  # TODO default strict?

            init_self(self, **kwargs)
            init_tree(
                self,
                name=name,
                parent=parent,
                children=children,
                dimensions=dimensions,
                coordinates=coordinates,
                strict=strict,
            )
            cls.__getattr__ = getattribute
            # TODO override __setattr__ for mutations

        cls.__init__ = init
        return cls

    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)


# TODO: add separate `component()` decorator like `attrs.field()`?
# for now, "bind" metadata indicates subcomponent, not a variable.
