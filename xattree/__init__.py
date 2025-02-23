"""
Herd an unruly glaring of `attrs` classes into an orderly `xarray.DataTree`.
"""

import builtins
import json
import types
from collections import ChainMap
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import datetime
from functools import singledispatch
from importlib.metadata import Distribution
from inspect import isclass
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    TypeVar,
    Union,
    dataclass_transform,
    get_args,
    get_origin,
    overload,
)

import attrs
import numpy as np
from beartype.claw import beartype_this_package
from beartype.vale import Is
from numpy.typing import ArrayLike, NDArray
from xarray import Dataset, DataTree

_PKG_URL = Distribution.from_name("xattree").read_text("direct_url.json")
_EDITABLE = json.loads(_PKG_URL).get("dir_info", {}).get("editable", False) if _PKG_URL else False
if _EDITABLE:
    beartype_this_package()


class DimsNotFound(KeyError):
    """Raised if an array variable specifies dimensions that can't be found."""

    pass


class CannotExpand(ValueError):
    """
    Raised if a scalar default is provided for an array variable
    specifying no dimensions. The scalar can't be expanded to an
    array without a known shape.
    """

    pass


_Int = int | np.int32 | np.int64
_Numeric = int | float | np.int64 | np.float64
_Scalar = bool | _Numeric | str | Path | datetime
_HasAttrs = Annotated[object, Is[lambda obj: attrs.has(type(obj))]]
_HasXats = Annotated[object, Is[lambda obj: xats(type(obj))]]

_CLAW = "host"
_SPEC = "spec"
_STRICT = "strict"
_WHERE = "where"
_WHERE_DEFAULT = "data"
_XATTREE_READY = "_xattree_ready"
_XATTREE_RESERVED_FIELDS = {
    "name": lambda cls: attrs.Attribute(
        name="name",
        default=cls.__name__.lower(),
        validator=None,
        repr=True,
        cmp=None,
        hash=True,
        eq=True,
        init=True,
        inherited=False,
        type=str,
    ),
    "dims": attrs.Attribute(
        name="dims",
        default=attrs.Factory(dict),
        validator=None,
        repr=False,
        cmp=None,
        hash=False,
        eq=False,
        init=True,
        inherited=False,
        type=Mapping[str, int],
    ),
    "parent": attrs.Attribute(
        name="parent",
        default=None,
        validator=None,
        repr=False,
        cmp=None,
        hash=False,
        eq=False,
        init=True,
        inherited=False,
        type=_HasAttrs,
    ),
    "strict": attrs.Attribute(
        name="strict",
        default=True,
        validator=None,
        repr=True,
        cmp=None,
        hash=False,
        eq=False,
        init=True,
        inherited=False,
        type=bool,
    ),
}


def _chexpand(value: ArrayLike, shape: tuple[int]) -> Optional[NDArray]:
    """If `value` is iterable, check its shape. If scalar, expand to shape."""
    value = np.array(value)
    if value.shape == ():
        return np.full(shape, value.item())
    if value.shape != shape:
        raise ValueError(f"Shape mismatch, got {value.shape}, expected {shape}")
    return value


@attrs.define
class _Xat:
    cls: Optional[type] = None
    name: Optional[str] = None
    attr: Optional[attrs.Attribute] = None
    optional: bool = False


@attrs.define
class _Dimension(_Xat):
    coord: Optional[str] = None
    scope: Optional[str] = None


@attrs.define
class _Coordinate(_Xat):
    dim: Optional[_Dimension] = None
    scope: Optional[str] = None


@attrs.define
class _Attribute(_Xat):
    pass


@attrs.define
class _Array(_Xat):
    dims: Optional[tuple[str, ...]] = None


@attrs.define
class _Child(_Xat):
    kind: Literal["one", "list", "dict"] = "one"


@attrs.define
class _XatSpec:
    dimensions: dict[str, _Dimension]
    coordinates: dict[str, _Coordinate]
    attributes: dict[str, _Attribute]
    children: dict[str, _Child]
    arrays: dict[str, _Array]

    @property
    def flat(self):
        """Flatten the specification into a single dict of fields."""
        return ChainMap(
            self.dimensions,
            self.coordinates,
            self.attributes,
            self.children,
            self.arrays,
        )


def _xat(attr: attrs.Attribute) -> Optional[_Xat]:
    """Extract a `Xattribute` from an `attrs.Attribute`."""
    if attr.type is None:
        raise TypeError(f"Field has no type: {attr.name}")
    if not xat(attr):
        return None

    type_ = attr.type
    args = get_args(type_)
    origin = get_origin(type_)
    var = attr.metadata.get(_SPEC)
    optional = var.optional

    match var:
        case _Dimension():
            if origin in (Union, types.UnionType):
                if args[-1] is types.NoneType:  # Optional
                    optional = True
                    type_ = args[0]
                else:
                    raise TypeError(f"Dim must have a concrete type: {attr.name}")
            if not (isclass(type_) and issubclass(type_, _Int)):
                raise TypeError(f"Dim '{attr.name}' must be an integer")
            return _Coordinate(
                name=var.coord or attr.name,
                dim=attrs.evolve(var, name=attr.name, attr=attr),
                scope=var.scope,
                attr=attr,
            )
        case _Coordinate():
            if not (isclass(origin) and issubclass(origin, np.ndarray)):
                raise TypeError(f"Coord '{attr.name}' must be an array type")
            return attrs.evolve(
                var,
                name=attr.name,
                dim=_Dimension(name=var.dim, attr=attr, scope=var.scope, coord=attr.name)
                if var.dim
                else _Dimension(name=attr.name, attr=attr),
                attr=attr,
            )
        case _Array():
            if origin in (Union, types.UnionType):
                if args[-1] is types.NoneType:  # Optional
                    optional = True
                    type_ = args[0]
                    if get_origin(type_) is np.ndarray:
                        origin = np.ndarray
                        type_ = get_args(type_)[1]
                    elif get_origin(type_) is list:
                        origin = list
                        type_ = get_args(type_)[0]
                    else:
                        origin = None
                else:
                    raise TypeError(f"Field must have a concrete type: {attr.name}")
            if not (isclass(origin) and issubclass(origin, (list, np.ndarray))):
                raise TypeError(f"Array '{attr.name}' type unsupported: {origin}")
            return attrs.evolve(var, name=attr.name, cls=type_, attr=attr, optional=optional)
        case _Attribute():
            if origin in (Union, types.UnionType):
                if args[-1] is types.NoneType:  # Optional
                    optional = True
                    type_ = args[0]
                else:
                    raise TypeError(f"Field must have a concrete type: {attr.name}")
            return attrs.evolve(var, name=attr.name, cls=type_, attr=attr, optional=optional)
        case _Child():
            if origin in (Union, types.UnionType):
                if args[-1] is types.NoneType:  # Optional
                    optional = True
                    origin = None
                    type_ = args[0]
            if not origin:
                kind = "one"
                if not attrs.has(type_):
                    raise TypeError(f"Child '{attr.name}' is not attrs: {type_}")
            else:
                if origin not in [list, dict]:
                    raise TypeError(f"Child collection '{attr.name}' must be a list or dictionary")
                match len(args):
                    case 1:
                        kind = "list"
                        type_ = args[0]
                        if not attrs.has(type_):
                            raise TypeError(f"List '{attr.name}' child type '{type_}' is not attrs")
                    case 2:
                        kind = "dict"
                        type_ = args[1]
                        if not (args[0] is str and attrs.has(type_)):
                            raise TypeError(f"Dict '{attr.name}' child type '{type_}' is not attrs")

            return attrs.evolve(
                var,
                name=attr.name,
                cls=type_,
                kind=kind,
                attr=attr,
                optional=optional,
            )

    raise TypeError(f"Field '{attr.name}' could not be classified as a dim, coord, scalar, array, or child variable")


@singledispatch
def _xatspec(arg) -> _XatSpec:
    raise NotImplementedError(
        f"Unsupported type '{type(arg)}' for xattree spec, pass `attrs` class or dict of `attrs.Attribute`."
    )


@_xatspec.register
def _(cls: type) -> _XatSpec:
    if not ((meta := getattr(cls, "__xattree__", None)) and (spec := meta.get(_SPEC, None))):
        return _xatspec(fields_dict(cls))
    return spec


@_xatspec.register
def _(fields: dict) -> _XatSpec:
    dimensions = {}
    coordinates = {}
    attributes = {}
    children = {}
    arrays = {}

    for field in fields.values():
        if field.name in _XATTREE_RESERVED_FIELDS.keys():
            continue
        match var := _xat(field):
            case _Coordinate():
                dimensions[var.dim.name] = var.dim
                coordinates[field.name] = var
            case _Array():
                arrays[field.name] = var
            case _Attribute():
                attributes[field.name] = var
            case _Child():
                children[field.name] = var
            case None:
                type_ = field.type
                origin = get_origin(type_)
                args = get_args(type_)
                optional = False
                if origin in (Union, types.UnionType):
                    if args[-1] is types.NoneType:  # Optional
                        optional = True
                        type_ = args[0]
                    else:
                        raise TypeError(f"Field may not be a union: {field.name}")
                iterable = isclass(origin) and issubclass(origin, Iterable)
                mapping = iterable and issubclass(origin, Mapping)
                if attrs.has(type_):
                    children[field.name] = _Child(
                        cls=type_,
                        name=field.name,
                        attr=field,
                        optional=optional,
                        kind="one",
                    )
                elif mapping and attrs.has(args[-1]):
                    children[field.name] = _Child(
                        cls=args[-1],
                        name=field.name,
                        attr=field,
                        optional=optional,
                        kind="dict",
                    )
                elif iterable and attrs.has(args[0]):
                    children[field.name] = _Child(
                        cls=args[-0],
                        name=field.name,
                        attr=field,
                        optional=optional,
                        kind="list",
                    )
                else:
                    attributes[field.name] = _Attribute(
                        cls=type_,
                        name=field.name,
                        attr=field,
                        optional=optional,
                    )

    return _XatSpec(
        dimensions=dimensions,
        coordinates=coordinates,
        attributes=attributes,
        children=children,
        arrays=arrays,
    )


def _bind_tree(
    self: _HasAttrs,
    parent: _HasAttrs = None,
    children: Optional[Mapping[str, _HasAttrs]] = None,
    where: str = _WHERE_DEFAULT,
):
    """Bind a tree to its parent and children."""
    name = getattr(self, where).name
    tree = getattr(self, where)
    children = children or {}

    # bind parent
    if parent:
        parent_tree = getattr(parent, where)
        if name in parent.data:
            parent_tree.update({name: tree})
        else:
            setattr(parent, where, parent_tree.assign({name: tree}))
        _climb(parent, parent_tree)
        parent_tree = getattr(parent, where)
        setattr(self, where, parent_tree[name])

        # self node will have been displaced
        # in parent since node child updates
        # don't happen in-place.
        tree = getattr(self, where)

    _climb(self, tree)

    # bind children
    for n, child in children.items():
        child_tree = getattr(child, where)
        _climb(child, tree[n])
        setattr(child, where, tree[n])
        _bind_tree(
            child, parent=self, children={n: c._cache[_CLAW] for n, c in child_tree.children.items()}, where=where
        )

    # give the data tree a reference to the instance
    # so it can be the class hierarchy's "backbone".
    _climb(self, tree)
    setattr(self, where, tree)


def _init_tree(self: _HasAttrs, strict: bool = True, where: str = _WHERE_DEFAULT):
    """
    Initialize a tree.

    Notes
    -----
    This method must run after the default `__init__()`.

    The tree is built from the class' `attrs` fields, i.e.
    spirited from the instance's `__dict__` into the tree,
    which is added as an attribute whose name is "where".
    `__dict__` is empty after this method runs except for
    the data tree. Field access is proxied to the tree.

    The class cannot use slots for this to work.
    """

    cls = type(self)
    cls_name = cls.__name__.lower()
    name = self.__dict__.pop("name", cls_name)
    parent = self.__dict__.pop("parent", None)
    xatspec = _xatspec(cls)
    dimensions = {}

    def _yield_children() -> Iterator[tuple[str, _HasAttrs]]:
        for field in xatspec.children.values():
            if (child := self.__dict__.pop(field.name, None)) is None:
                continue
            match field.kind:
                case "one":
                    yield (field.name, child)
                case "list":
                    for i, c in enumerate(child):
                        yield (f"{field.name}_{i}", c)
                case "dict":
                    for k, c in child.items():
                        yield (k, c)
                case _:
                    raise TypeError(f"Bad child collection field '{field.name}'")

    def _yield_attrs() -> Iterator[tuple[str, Any]]:
        for field in xatspec.attributes.values():
            yield (field.name, self.__dict__.pop(field.name, field.attr.default))

    children = dict(list(_yield_children()))
    attributes = dict(list(_yield_attrs()))

    def _resolve_array(
        field: _Xat,
        value: ArrayLike,
        strict: bool = False,
        **dimensions,
    ) -> tuple[Optional[NDArray], Optional[dict[str, NDArray]]]:
        dimensions = dimensions or {}
        match field:
            case _Coordinate():
                if field.dim.attr.default is None:
                    raise CannotExpand(
                        f"Class '{cls_name}' coord array '{field.name}'"
                        f"paired with dim '{field.dim.name}' can't expand "
                        f"without a default dimension value."
                    )
                return _chexpand(value, (field.dim.attr.default,))
            case _Array():
                if field.dims is None and (value is None or isinstance(value, _Scalar)):
                    raise CannotExpand(f"Class '{cls_name}' array '{field.name}' can't expand, no dims.")
                if value is None:
                    value = field.attr.default
                if field.dims is None:
                    return value
                shape = tuple([dimensions.pop(dim, dim) for dim in field.dims])
                unresolved = [dim for dim in shape if not isinstance(dim, int)]
                if any(unresolved):
                    if strict:
                        raise DimsNotFound(
                            f"Class '{cls_name}' array '{field.name}' failed dim resolution: {', '.join(unresolved)}"
                        )
                    return None
                return _chexpand(value, shape)

    def _yield_coords(scope: str) -> Iterator[tuple[str, tuple[str, Any]]]:
        # inherit coordinates from parent.. necessary or happens automatically?
        if parent:
            parent_tree = getattr(parent, where)
            for coord_name, coord in parent_tree.coords.items():
                dim_name = coord.dims[0]
                dimensions[dim_name] = coord.data.size
                yield (coord_name, (dim_name, coord.data))

        # find self-scoped coordinates defined in children
        # TODO: terrible hack, only works one level down,
        # need to register any declared dims/coords at
        # decoration time and look up by path in child
        for obj in children.values():
            child_type = type(obj)
            child_origin = get_origin(child_type)
            child_args = get_args(child_type)
            is_iterable = child_origin and issubclass(child_origin, Iterable)
            if is_iterable:
                child_type = child_args[0]
            if not attrs.has(child_type):
                continue
            spec = _xatspec(child_type)
            tree = getattr(obj, where)
            for var in spec.dimensions.values():
                if scope == var.scope:
                    coord = var.coord
                    coord_arr = tree.coords[coord].data
                    dimensions[var.name] = coord_arr.size
                    yield coord, (var.name, coord_arr)
            for var in spec.coordinates.values():
                if scope == var.scope:
                    dim = var.dim
                    coord_arr = tree.coords[var.name].data
                    dimensions[dim.name] = coord_arr.size
                    yield var.name, (dim.name, coord_arr)

        for var in xatspec.coordinates.values():
            dim_name = var.dim.name if var.dim else var.name
            if (array := self.__dict__.pop(var.name, None)) is not None:
                yield (var.name, (dim_name, array))
            if not (stop := self.__dict__.pop(dim_name, None)):
                if not (stop := var.attr.default):
                    continue
            match type(stop):
                case builtins.int | np.int64:
                    step = 1
                    start = 0
                case builtins.float | np.float64:
                    step = 1.0
                    start = 0.0
                case _:
                    raise ValueError("Dimensions/coordinates must be numeric.")
            array = np.arange(start, stop, step)
            dimensions[dim_name] = array.size
            yield (var.name, (dim_name, array))

    # resolve dimensions/coordinates before arrays
    coordinates = dict(list(_yield_coords(scope=cls_name)))

    def _yield_arrays() -> Iterator[tuple[str, ArrayLike | tuple[str, ArrayLike]]]:
        explicit_dims = self.__dict__.pop("dims", None) or {}
        for var in xatspec.arrays.values():
            if (
                array := _resolve_array(
                    var,
                    value=self.__dict__.pop(var.name, var.attr.default),
                    strict=strict,
                    **dimensions | explicit_dims,
                )
            ) is not None and var.attr.default is not None:
                yield (var.name, (var.dims, array) if var.dims else array)

    setattr(
        self,
        where,
        _climb(
            has_xats=self,
            tree=DataTree(
                dataset=Dataset(
                    data_vars=dict(list(_yield_arrays())),
                    coords=coordinates,
                    attrs={n: v for n, v in attributes.items()},
                ),
                name=name,
                children={n: getattr(c, where) for n, c in children.items()},
            ),
        ),
    )
    _bind_tree(self, parent=parent, children=children)


def _getattribute(self: _HasAttrs, name: str) -> Any:
    cls = type(self)
    if name == (where := cls.__xattree__[_WHERE]):
        raise AttributeError
    if name == _XATTREE_READY:
        return False
    tree: DataTree = getattr(self, where, None)
    match name:
        case "name":
            return tree.name
        case "dims":
            return tree.dims
        case "parent":
            return None if tree.is_root else tree.parent._cache[_CLAW]
        case "children":
            # TODO: make `children` a full-fledged attribute?
            return {n: c._cache[_CLAW] for n, c in tree.children.items()}
    spec = _xatspec(cls)
    if field := spec.flat.get(name, None):
        match field:
            case _Dimension():
                return tree.dims[field.name]
            case _Coordinate():
                return tree.coords[field.name].data
            case _Attribute():
                return tree.attrs[field.name]
            case _Array():
                try:
                    return tree[field.name]
                except KeyError:
                    return None
            case _Child():
                if field.kind == "dict":
                    return {
                        n: c._cache[_CLAW]
                        for n, c in tree.children.items()
                        if issubclass(type(c._cache[_CLAW]), field.cls)
                    }
                if field.kind == "list":
                    return [
                        c._cache[_CLAW] for c in tree.children.values() if issubclass(type(c._cache[_CLAW]), field.cls)
                    ]
                if field.kind == "one":
                    return next(
                        (
                            c._cache[_CLAW]
                            for c in tree.children.values()
                            if c.name == field.name and issubclass(type(c._cache[_CLAW]), field.cls)
                        ),
                        None,
                    )
            case _:
                raise TypeError(f"Field '{name}' is not a dimension, coordinate, attribute, array, or child variable")

    raise AttributeError


def _setattribute(self: _HasAttrs, name: str, value: Any):
    cls = type(self)
    cls_name = cls.__name__
    where = cls.__xattree__[_WHERE]
    if not getattr(self, _XATTREE_READY, False) or name in [
        where,
        _XATTREE_READY,
    ]:
        self.__dict__[name] = value
        return
    spec = _xatspec(cls)
    if not (field := spec.flat.get(name, None)):
        raise AttributeError(f"{cls_name} has no attribute {name}")
    match field:
        case _Dimension():
            raise AttributeError(f"Cannot set dimension '{name}'.")
        case _Coordinate():
            raise AttributeError(f"Cannot set coordinate '{name}'.")
        case _Attribute():
            self.data.attrs[field.name] = value
        case _Array():
            self.data.update({field.name: value})
        case _Child():
            _bind_tree(
                self,
                children=self.children | {field.name: getattr(value, where)._cache[_CLAW]},
            )


def dim(
    coord,
    scope=None,
    default=attrs.NOTHING,
    validator=None,
    repr=True,
    eq=True,
    metadata=None,
):
    """Create a dimension field."""
    metadata = metadata or {}
    metadata[_SPEC] = _Dimension(coord=coord, scope=scope)
    return attrs.field(
        default=default,
        validator=validator,
        repr=repr,
        eq=eq,
        order=False,
        hash=True,
        init=True,
        metadata=metadata,
    )


def coord(
    dim=None,
    scope=None,
    default=attrs.NOTHING,
    validator=None,
    repr=True,
    eq=True,
    metadata=None,
):
    """Create a coordinate field."""
    metadata = metadata or {}
    metadata[_SPEC] = _Coordinate(dim=dim, scope=scope)
    return attrs.field(
        default=default,
        validator=validator,
        repr=repr,
        eq=eq,
        order=False,
        hash=True,
        init=True,
        metadata=metadata,
    )


def array(
    cls=None,
    dims=None,
    default=attrs.NOTHING,
    validator=None,
    repr=True,
    eq=None,
    metadata=None,
):
    """Create an array field."""

    dims = dims if isinstance(dims, Iterable) else tuple()
    if not any(dims) and isinstance(default, _Scalar):
        raise CannotExpand("If no dims, no scalar defaults.")
    if cls and default is attrs.NOTHING:
        default = attrs.Factory(cls)
    metadata = metadata or {}
    metadata[_SPEC] = _Array(cls=cls, dims=dims)
    return attrs.field(
        default=default,
        validator=validator,
        repr=repr,
        eq=eq or attrs.cmp_using(eq=np.array_equal),
        order=False,
        hash=False,
        init=True,
        metadata=metadata,
    )


def xat(field: attrs.Attribute) -> bool:
    """Check whether `field` is a `xattree` field."""
    return _SPEC in field.metadata


def xats(cls) -> bool:
    """Check whether `cls` is a `xattree`."""
    if not getattr(cls, "__xattree__", None):
        return False
    # TODO any validation?
    return True


def fields_dict(cls, just_yours: bool = True) -> dict[str, attrs.Attribute]:
    """
    Get the `attrs` fields of a class. By default, only your
    attributes are returned, none of the special attributes
    attached by `xattree`. To include those attributes, set
    `just_yours=False`.
    """
    return {
        n: f for n, f in attrs.fields_dict(cls).items() if not just_yours or n not in _XATTREE_RESERVED_FIELDS.keys()
    }


def fields(cls, just_yours: bool = True) -> list[attrs.Attribute]:
    """
    Get the `attrs` fields of a class. By default, only your
    attributes are returned, none of the special attributes
    attached by `xattree`. To include those attributes, set
    `just_yours=False`.
    """
    return list(fields_dict(cls, just_yours).values())


def _climb(has_xats: _HasXats, tree: DataTree) -> DataTree:
    """
    Like a cat or the toxoplasmosis it carries, the infected instance extends a
    claw into the victim('s `_cache`) and hijacks its body for its own purposes.
    """
    try:
        tree._cache[_CLAW] = has_xats
    except AttributeError:
        tree._cache = {_CLAW: has_xats}
    return tree


T = TypeVar("T")


@overload
def xattree(
    *,
    where: str = _WHERE_DEFAULT,
) -> Callable[[type[T]], type[T]]: ...


@overload
def xattree(maybe_cls: type[T]) -> type[T]: ...


@dataclass_transform(field_specifiers=(attrs.field, dim, coord, array))
def xattree(
    maybe_cls: Optional[type[_HasAttrs]] = None,
    *,
    where: str = _WHERE_DEFAULT,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Make an `attrs`-based class a (node in a) cat tree."""

    def wrap(cls):
        orig_pre_init = getattr(cls, "__attrs_pre_init__", None)
        orig_post_init = getattr(cls, "__attrs_post_init__", None)

        def pre_init(self):
            if orig_pre_init:
                orig_pre_init(self)
            setattr(self, _XATTREE_READY, False)

        def post_init(self):
            if orig_post_init:
                orig_post_init(self)
            _init_tree(self, strict=self.strict, where=cls.__xattree__[_WHERE])
            setattr(self, _XATTREE_READY, True)

        def transformer(cls: type, fields: list[attrs.Attribute]) -> Iterator[attrs.Attribute]:
            def _transform_field(field):
                type_ = field.type
                args = get_args(type_)
                origin = get_origin(type_)
                iterable = isclass(origin) and issubclass(origin, Iterable)
                mapping = iterable and issubclass(origin, Mapping)
                if not (attrs.has(type_) or (mapping and attrs.has(args[-1])) or (iterable and attrs.has(args[0]))):
                    return field
                metadata = field.metadata.copy() or {}
                metadata[_SPEC] = _Child(
                    cls=type_,
                    name=field.name,
                    attr=field,
                    kind="dict" if mapping else "list" if iterable else "one",
                )
                default = field.default
                if default is attrs.NOTHING:
                    default = attrs.Factory(lambda: type_(**({} if iterable else {_STRICT: False})))
                elif default is None and iterable:
                    raise ValueError("Child collection's default may not be None.")
                return attrs.Attribute(
                    name=field.name,
                    default=default,
                    validator=field.validator,
                    repr=field.repr,
                    cmp=None,
                    hash=field.hash,
                    eq=field.eq,
                    init=field.init,
                    inherited=field.inherited,
                    metadata=field.metadata,
                    type=field.type,
                    converter=field.converter,
                    kw_only=field.kw_only,
                    eq_key=field.eq_key,
                    order=field.order,
                    order_key=field.order_key,
                    on_setattr=field.on_setattr,
                    alias=field.alias,
                )

            attrs_ = [_transform_field(f) for f in fields]
            xattrs = [f(cls) if isinstance(f, Callable) else f for f in _XATTREE_RESERVED_FIELDS.values()]
            return attrs_ + xattrs

        cls.__attrs_pre_init__ = pre_init
        cls.__attrs_post_init__ = post_init
        cls = attrs.define(cls, slots=False, field_transformer=transformer)
        cls.__getattr__ = _getattribute
        cls.__setattr__ = _setattribute
        cls.__xattree__ = {_WHERE: where, _SPEC: _xatspec(cls)}
        return cls

    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)
