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
from itertools import chain
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
import xarray as xa
from beartype.claw import beartype_this_package
from beartype.vale import Is
from numpy.typing import ArrayLike, NDArray

_PKG_NAME = "xattree"
_PKG_URL = Distribution.from_name(_PKG_NAME).read_text("direct_url.json")
_EDITABLE = json.loads(_PKG_URL).get("dir_info", {}).get("editable", False) if _PKG_URL else False
if _EDITABLE:
    beartype_this_package()


class Xattree(xa.DataTree):
    # DataTree is not yet a proper slotted class, it still has `__dict__`.
    # So monkey-patching is not strictly necessary yet, but it will be.
    # When it is, this will start enforcing no dynamic attributes. See
    #   - https://github.com/pydata/xarray/issues/9068
    #   - https://github.com/pydata/xarray/issues/9928
    __slots__ = ("_host",)

    def __init__(self, dataset=None, children=None, name=None, host=None):
        super().__init__(dataset=dataset, children=children, name=name)
        self._host = host


xa.DataTree = Xattree


class DimsNotFound(KeyError):
    """Raised if an array field specifies dimensions that can't be found."""

    pass


class CannotExpand(ValueError):
    """
    Raised if a scalar default is provided for an array field
    specifying no dimensions. The scalar can't be expanded to an
    array without a known shape.
    """

    pass


_Int = int | np.int32 | np.int64
_Numeric = int | float | np.int64 | np.float64
_Scalar = bool | _Numeric | str | Path | datetime
_HasAttrs = Annotated[object, Is[lambda obj: attrs.has(type(obj))]]
_HasXats = Annotated[object, Is[lambda obj: has_xats(type(obj))]]

_SPEC = "spec"
_STRICT = "strict"
_WHERE = "where"
_WHERE_DEFAULT = "data"
_XATTREE_DUNDER = "__xattree__"
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
    """A `xattree` attribute (field) specification."""

    cls: Optional[type] = None
    name: Optional[str] = None
    attr: Optional[attrs.Attribute] = None
    optional: bool = False


@attrs.define
class _Dimension(_Xat):
    """Dimension field specification."""

    scope: Optional[str] = None


@attrs.define
class _Coordinate(_Xat):
    """Coordinate field specification."""

    dim: Optional[_Dimension] = None
    scope: Optional[str] = None


@attrs.define
class _Attribute(_Xat):
    """Arbitrary attribute field specification."""

    pass


@attrs.define
class _Array(_Xat):
    """Array field specification."""

    dims: Optional[tuple[str, ...]] = None


@attrs.define
class _Child(_Xat):
    """Child field specification."""

    kind: Literal["one", "list", "dict"] = "one"


@attrs.define
class _XatSpec:
    """A `xattree`-decorated class specification."""

    coordinates: dict[str, _Coordinate]
    dimensions: dict[str, _Dimension]
    attributes: dict[str, _Attribute]
    children: dict[str, _Child]
    arrays: dict[str, _Array]

    @property
    def flat(self) -> Mapping[str, _Xat]:
        """Flatten the specification into a single dict of fields."""
        return ChainMap(
            self.coordinates,
            self.dimensions,
            self.attributes,
            self.children,
            self.arrays,
        )


def _extrixate(attr: attrs.Attribute) -> Optional[_Xat]:
    """
    Extricate a full `_Xat`(-tribute) specification from an `attrs.Attribute`
    with `xattree` metadata. Gets used to build the `xattree` specification.

    The `dim()`, `coord()`, `array()` and `child()` decorators put `_Xat`s
    in the metadata of the `attrs` fields they decorate, but these are not
    fully initialized (e.g. we don't know the `cls` of a `child` field yet).
    This function gets a partially initialized `_Xat` from the `Attribute`
    metadata, finishes initializing it, and returns it.

    Note that xarray doesn't allow dimensions to live separately from their
    coordinate arrays, but we allow `xattree` classes to define dimensions,
    then infer coordinates from them at the time the `_Xatspec` is created,
    so we have separate concepts/abstractions.
    """
    if attr.type is None:
        raise TypeError(f"Field has no type: {attr.name}")
    if not is_xat(attr):
        return None

    type_ = attr.type
    args = get_args(type_)
    origin = get_origin(type_)
    spec = attr.metadata[_SPEC]
    optional = spec.optional

    match spec:
        case _Dimension():
            if origin in (Union, types.UnionType):
                if args[-1] is types.NoneType:  # Optional
                    optional = True
                    type_ = args[0]
                else:
                    raise TypeError(f"Dim must have a concrete type: {attr.name}")
            if not (isclass(type_) and issubclass(type_, _Int)):
                raise TypeError(f"Dim '{attr.name}' must be an integer")
            return attrs.evolve(
                spec,
                name=spec.name or attr.name,
                attr=attr,
                scope=spec.scope,
                optional=optional,
            )
        case _Coordinate():
            if not (isclass(origin) and issubclass(origin, np.ndarray)):
                raise TypeError(f"Coord '{attr.name}' must be an array type")
            return attrs.evolve(
                spec,
                name=attr.name,
                dim=_Dimension(name=spec.name or attr.name, attr=attr, scope=spec.scope),
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
            return attrs.evolve(spec, name=attr.name, cls=type_, attr=attr, optional=optional)
        case _Attribute():
            if origin in (Union, types.UnionType):
                if args[-1] is types.NoneType:  # Optional
                    optional = True
                    type_ = args[0]
                else:
                    raise TypeError(f"Field must have a concrete type: {attr.name}")
            return attrs.evolve(spec, name=attr.name, cls=type_, attr=attr, optional=optional)
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
                spec,
                name=attr.name,
                cls=type_,
                kind=kind,
                attr=attr,
                optional=optional,
            )

    raise TypeError(
        f"Field '{attr.name}' could not be classified as a dim, "
        "coord, scalar, array, or child variable"
    )


@singledispatch
def _xatspec(arg) -> _XatSpec:
    raise NotImplementedError(
        f"Unsupported type '{type(arg)}' for xattree spec, "
        f"pass `attrs` class or dict of `attrs.Attribute`."
    )


@_xatspec.register
def _(cls: type) -> _XatSpec:
    if not ((meta := getattr(cls, _XATTREE_DUNDER, None)) and (spec := meta.get(_SPEC, None))):
        return _xatspec(fields_dict(cls))
    return spec


@_xatspec.register
def _(fields: dict) -> _XatSpec:
    coordinates = {}
    dimensions = {}
    attributes = {}
    children = {}
    arrays = {}

    for field in fields.values():
        if field.name in _XATTREE_RESERVED_FIELDS.keys():
            continue
        match xat := _extrixate(field):
            case _Dimension():
                dimensions[field.name] = xat
            case _Coordinate():
                coordinates[xat.name] = xat
            case _Array():
                arrays[field.name] = xat
            case _Attribute():
                attributes[field.name] = xat
            case _Child():
                children[field.name] = xat
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
        coordinates=coordinates,
        dimensions=dimensions,
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
    """
    Bind a tree to its parent and children, and give each tree node
    a reference to its host.
    """
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
        parent_tree._host = parent
        parent_tree = getattr(parent, where)
        setattr(self, where, parent_tree[name])
        # self node will have been displaced
        # in parent since node child updates
        # don't happen in-place.
        tree = getattr(self, where)

    # bind children
    for n, child in children.items():
        child_tree = getattr(child, where)
        tree[n]._host = child
        setattr(child, where, tree[n])
        _bind_tree(
            child,
            parent=self,
            children={n: c._host for n, c in child_tree.children.items()},
            where=where,
        )

    tree._host = self
    setattr(self, where, tree)


def _init_tree(self: _HasAttrs, strict: bool = True, where: str = _WHERE_DEFAULT):
    """
    Initialize a `xattree`-decorated class instance's `DataTree`.

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
    cls_name = cls.__name__
    name = self.__dict__.pop("name", cls_name.lower())
    parent = self.__dict__.pop("parent", None)
    xatspec = _xatspec(cls)
    dimensions = {}

    def _yield_children() -> Iterator[tuple[str, _HasAttrs]]:
        for xat in xatspec.children.values():
            if (child := self.__dict__.pop(xat.name, None)) is None:
                continue
            match xat.kind:
                case "one":
                    yield (xat.name, child)
                case "list":
                    for i, c in enumerate(child):
                        yield (f"{xat.name}_{i}", c)
                case "dict":
                    for k, c in child.items():
                        yield (k, c)
                case _:
                    raise TypeError(f"Bad child collection field '{xat.name}'")

    def _yield_attrs() -> Iterator[tuple[str, Any]]:
        for xat_name, xat in xatspec.dimensions.items():
            yield (xat_name, self.__dict__.get(xat_name, xat.attr.default))
        for xat_name, xat in xatspec.attributes.items():
            yield (xat_name, self.__dict__.pop(xat_name, xat.attr.default))

    children = dict(list(_yield_children()))
    attributes = dict(list(_yield_attrs()))

    def _resolve_array(
        xat: _Xat, value: ArrayLike, strict: bool = False, **dims
    ) -> Optional[NDArray]:
        dims = dims or {}
        match xat:
            case _Coordinate():
                if xat.dim.attr.default is None:
                    raise CannotExpand(
                        f"Class '{cls_name}' coord array '{xat.name}'"
                        f"paired with dim '{xat.dim.name}' can't expand "
                        f"without a default dimension size."
                    )
                return _chexpand(value, (xat.dim.attr.default,))
            case _Array():
                if xat.dims is None and (value is None or isinstance(value, _Scalar)):
                    raise CannotExpand(
                        f"Class '{cls_name}' array '{xat.name}' can't expand "
                        "without explicit dimensions or a non-scalar default."
                    )
                if value is None:
                    value = xat.attr.default
                if xat.dims is None:
                    return value
                shape = tuple([dims.pop(dim, dim) for dim in xat.dims])
                unresolved = [dim for dim in shape if not isinstance(dim, int)]
                if any(unresolved):
                    if strict:
                        raise DimsNotFound(
                            f"Class '{cls_name}' array '{xat.name}' "
                            f"failed dim resolution: {', '.join(unresolved)}"
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
        for child in children.values():
            child_type = type(child)
            child_origin = get_origin(child_type)
            child_args = get_args(child_type)
            is_iterable = child_origin and issubclass(child_origin, Iterable)
            if is_iterable:
                child_type = child_args[0]
            if not attrs.has(child_type):
                continue
            spec = _xatspec(child_type)
            tree = getattr(child, where)
            for xat in spec.dimensions.values():
                if scope == xat.scope:
                    coord_arr = tree.coords[xat.name].data
                    dimensions[xat.name] = coord_arr.size
                    yield xat.name, (xat.name, coord_arr)

        for xat in chain(xatspec.coordinates.values(), xatspec.dimensions.values()):
            value = self.__dict__.pop(
                xat.attr.name if xat.attr.name != xat.name else xat.name, None
            )
            if value is None or value is attrs.NOTHING:
                value = xat.attr.default
            if value is None or value is attrs.NOTHING:
                continue
            match xat:
                case _Coordinate():
                    dimensions[xat.name] = len(value)
                    yield (xat.name, (xat.name, value))
                    continue
                case _Dimension():
                    if isinstance(value, (builtins.int, np.int64)):
                        step = 1
                        start = 0
                    elif isinstance(value, (builtins.float | np.float64)):
                        step = 1.0
                        start = 0.0
                    else:
                        raise ValueError("Dimension bounds must be numeric.")
                    array = np.arange(start, value, step)
                    dimensions[xat.name] = array.size
                    yield (xat.name, (xat.name, array))

    # resolve dimensions/coordinates before arrays
    coordinates = dict(list(_yield_coords(scope=cls_name.lower())))

    def _yield_arrays() -> Iterator[tuple[str, ArrayLike | tuple[str, ArrayLike]]]:
        explicit_dims = self.__dict__.pop("dims", None) or {}
        for xat in xatspec.arrays.values():
            if (
                array := _resolve_array(
                    xat,
                    value=self.__dict__.pop(xat.name, xat.attr.default),
                    strict=strict,
                    **dimensions | explicit_dims,
                )
            ) is not None and xat.attr.default is not None:
                yield (xat.name, (xat.dims, array) if xat.dims else array)

    arrays = dict(list(_yield_arrays()))

    setattr(
        self,
        where,
        Xattree(
            dataset=xa.Dataset(
                data_vars=arrays,
                coords=coordinates,
                attrs={n: a for n, a in attributes.items()},
            ),
            name=name,
            children={n: getattr(c, where) for n, c in children.items()},
            host=self,
        ),
    )
    _bind_tree(self, parent=parent, children=children)


def _getattribute(self: _HasAttrs, name: str) -> Any:
    cls = type(self)
    if name == (where := cls.__xattree__[_WHERE]):
        raise AttributeError
    if name == _XATTREE_READY:
        return False
    tree: xa.DataTree = getattr(self, where, None)
    match name:
        case "name":
            return tree.name
        case "dims":
            return tree.dims
        case "parent":
            return None if tree.is_root else tree.parent._host
        case "children":
            # TODO: make `children` a full-fledged attribute?
            return {n: c._host for n, c in tree.children.items()}
    spec = _xatspec(cls)
    if xat := spec.flat.get(name, None):
        match xat:
            case _Dimension():
                try:
                    return tree.dims[xat.name]
                except KeyError:
                    return tree.attrs[xat.name]
            case _Coordinate():
                return tree.coords[xat.name].data
            case _Attribute():
                return tree.attrs[xat.name]
            case _Array():
                try:
                    return tree[xat.name]
                except KeyError:
                    # TODO shouldn't do this?
                    return None
            case _Child():
                if xat.kind == "dict":
                    return {
                        n: c._host
                        for n, c in tree.children.items()
                        if issubclass(type(c._host), xat.cls)
                    }
                if xat.kind == "list":
                    return [
                        c._host
                        for c in tree.children.values()
                        if issubclass(type(c._host), xat.cls)
                    ]
                if xat.kind == "one":
                    return next(
                        (
                            c._host
                            for c in tree.children.values()
                            if c.name == xat.name and issubclass(type(c._host), xat.cls)
                        ),
                        None,
                    )
            case _:
                raise TypeError(
                    f"Field '{name}' is not a dimension, coordinate, "
                    "attribute, array, or child variable"
                )

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
    if not (xat := spec.flat.get(name, None)):
        raise AttributeError(f"{cls_name} has no field {name}")
    match xat:
        case _Dimension():
            raise AttributeError(f"Cannot set dimension '{name}'.")
        case _Coordinate():
            raise AttributeError(f"Cannot set coordinate '{name}'.")
        case _Attribute():
            self.data.attrs[xat.name] = value
        case _Array():
            self.data.update({xat.name: value})
        case _Child():
            _bind_tree(
                self,
                children=self.children | {xat.name: getattr(value, where)._host},
            )


def dim(
    name=None,
    scope=None,
    default=attrs.NOTHING,
    validator=None,
    repr=True,
    eq=True,
    metadata=None,
):
    """Create a dimension field."""
    metadata = metadata or {}
    metadata[_SPEC] = _Dimension(name=name, scope=scope)
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
    scope=None,
    default=attrs.NOTHING,
    validator=None,
    repr=True,
    eq=True,
    metadata=None,
):
    """Create a coordinate field."""
    metadata = metadata or {}
    metadata[_SPEC] = _Coordinate(scope=scope)
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


def is_xat(field: attrs.Attribute) -> bool:
    """Check whether `field` is a `xattree` field."""
    return _SPEC in field.metadata


def has_xats(cls) -> bool:
    """Check whether `cls` is a `xattree`."""
    if not getattr(cls, _XATTREE_DUNDER, None):
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
        n: f
        for n, f in attrs.fields_dict(cls).items()
        if not just_yours or n not in _XATTREE_RESERVED_FIELDS.keys()
    }


def fields(cls, just_yours: bool = True) -> list[attrs.Attribute]:
    """
    Get the `attrs` fields of a class. By default, only your
    attributes are returned, none of the special attributes
    attached by `xattree`. To include those attributes, set
    `just_yours=False`.
    """
    return list(fields_dict(cls, just_yours).values())


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
    """Make an `attrs`-based class a (node in a) `xattree`."""

    def wrap(cls):
        if has_xats(cls):
            raise TypeError("Class is already a `xattree`.")

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
                if field.name in _XATTREE_RESERVED_FIELDS.keys():
                    raise ValueError(f"Field name '{field.name}' is reserved.")

                type_ = field.type
                args = get_args(type_)
                origin = get_origin(type_)
                iterable = isclass(origin) and issubclass(origin, Iterable)
                mapping = iterable and issubclass(origin, Mapping)
                if not (
                    attrs.has(type_)
                    or (mapping and attrs.has(args[-1]))
                    or (iterable and attrs.has(args[0]))
                ):
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
            xattrs = [
                f(cls) if isinstance(f, Callable) else f for f in _XATTREE_RESERVED_FIELDS.values()
            ]
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
