"""
Herd an unruly glaring of `attrs` classes into an orderly `xarray.DataTree`.
"""

import builtins
import json
import types
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import datetime
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
from xarray import DataArray, Dataset, DataTree

_PKG_URL = Distribution.from_name("xattree").read_text("direct_url.json")
_EDITABLE = (
    json.loads(_PKG_URL).get("dir_info", {}).get("editable", False)
    if _PKG_URL
    else False
)
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


DIM = "dim"
DIMS = "dims"
NAME = "name"
ARRAY = "array"
COORD = "coord"
CHILD = "child"
SCOPE = "scope"
STRICT = "strict"
SPEC = "spec"
TYPE = "type"
_WHERE = "where"
_WHERE_DEFAULT = "data"
_READY = "ready"
_XATTREE_READY = "_xattree_ready"
_XATTREE_FIELDS = {
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
        raise ValueError(
            f"Shape mismatch, got {value.shape}, expected {shape}"
        )
    return value


def _get(
    tree: DataTree, key: str, default: Optional[Any] = None
) -> Optional[Any]:
    """
    Get a scalar (dimension or attribute) or array value from `tree`.
    Needed because `DataTree.get()` doesn't look in `dims` or `attrs`.
    """
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


@attrs.define
class _Xattribute:
    cls: Optional[type] = None
    name: Optional[str] = None
    attr: Optional[attrs.Attribute] = None
    optional: bool = False


@attrs.define
class _Dimension(_Xattribute):
    coord: Optional[str] = None
    scope: Optional[str] = None


@attrs.define
class _Coordinate(_Xattribute):
    dim: Optional[_Dimension] = None
    scope: Optional[str] = None


@attrs.define
class _Attribute(_Xattribute):
    pass


@attrs.define
class _Array(_Xattribute):
    dims: Optional[tuple[str, ...]] = None


@attrs.define
class _Child(_Xattribute):
    kind: Literal["one", "list", "dict"] = "one"


@attrs.define
class _Xattree:
    dimensions: dict[str, _Dimension]
    coordinates: dict[str, _Coordinate]
    attributes: dict[str, _Attribute]
    children: dict[str, _Child]
    arrays: dict[str, _Array]


def _var_spec(attr: attrs.Attribute) -> Optional[_Xattribute]:
    """Extract a `Xattribute` from an `attrs.Attribute`."""

    if attr.type is None:
        raise TypeError(f"Field has no type: {attr.name}")

    if not _xattr(attr):
        return None

    type_ = attr.type
    args = get_args(type_)
    origin = get_origin(type_)
    var = attr.metadata.get(SPEC)
    optional = var.optional

    match var:
        case _Dimension():
            if origin in (Union, types.UnionType):
                if args[-1] is types.NoneType:  # Optional
                    optional = True
                    type_ = args[0]
                else:
                    raise TypeError(
                        f"Dim must have a concrete type: {attr.name}"
                    )
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
                dim=_Dimension(
                    name=var.dim, attr=attr, scope=var.scope, coord=attr.name
                )
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
                    else:
                        origin = None
                else:
                    raise TypeError(
                        f"Field must have a concrete type: {attr.name}"
                    )
            if not (isclass(origin) and issubclass(origin, np.ndarray)):
                raise TypeError(
                    f"Array '{attr.name}' type unsupported: {origin}"
                )
            return attrs.evolve(
                var, name=attr.name, cls=type_, attr=attr, optional=optional
            )
        case _Attribute():
            if origin in (Union, types.UnionType):
                if args[-1] is types.NoneType:  # Optional
                    optional = True
                    type_ = args[0]
                else:
                    raise TypeError(
                        f"Field must have a concrete type: {attr.name}"
                    )
            return attrs.evolve(
                var, name=attr.name, cls=type_, attr=attr, optional=optional
            )
        case _Child():
            if origin in (Union, types.UnionType):
                if args[-1] is types.NoneType:  # Optional
                    optional = True
                    origin = None
                    type_ = args[0]
            if not origin:
                kind = "one"
                if not attrs.has(type_):
                    raise TypeError(
                        f"Child '{attr.name}' is not attrs: {type_}"
                    )
            else:
                if origin not in [list, dict]:
                    raise TypeError(
                        f"Child collection '{attr.name}' "
                        f"must be a list or dictionary"
                    )
                match len(args):
                    case 1:
                        kind = "list"
                        type_ = args[0]
                        if not attrs.has(type_):
                            raise TypeError(
                                f"List '{attr.name}' child "
                                f"type '{type_}' is not attrs"
                            )
                    case 2:
                        kind = "dict"
                        type_ = args[1]
                        if not (args[0] is str and attrs.has(type_)):
                            raise TypeError(
                                f"Dict '{attr.name}' child "
                                f"type '{type_}' is not attrs"
                            )

            return attrs.evolve(
                var,
                name=attr.name,
                cls=type_,
                kind=kind,
                attr=attr,
                optional=optional,
            )

    raise TypeError(
        f"Field '{attr.name}' could not be classified as "
        f"a dim, coord, scalar, array, or child variable"
    )


def _xattrs_spec(fields: Mapping[str, attrs.Attribute]) -> _Xattree:
    """Parse a cat tree specification from an `attrs` class specification."""
    dimensions = {}
    coordinates = {}
    attributes = {}
    children = {}
    arrays = {}

    for field in fields.values():
        if field.name in _XATTREE_FIELDS.keys():
            continue
        match var := _var_spec(field):
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
                        raise TypeError(
                            f"Field may not be a union: {field.name}"
                        )
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

    return _Xattree(
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
        parent_tree.self = parent

        parent_tree = getattr(parent, where)
        setattr(self, where, parent_tree[name])

        # self node will have been displaced
        # in parent since node child updates
        # don't happen in-place.
        tree = getattr(self, where)

    tree.self = self

    # bind children
    for n, child in children.items():
        tree[n].self = child
        setattr(child, where, tree[n])

    # give the data tree a reference to the instance
    # so it can be the class hierarchy's "backbone",
    # i.e. so that an instance can be accessed from
    # another instance's data tree in `getattribute`.
    # TODO: think thru the consequences here. how to
    # avoid memory leaks?
    tree.self = self
    setattr(self, where, tree)


def _init_tree(
    self: _HasAttrs, strict: bool = True, where: str = _WHERE_DEFAULT
):
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
    xatspec = _xattrs_spec(fields_dict(cls))
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
                    raise TypeError(
                        f"Bad child collection field '{field.name}'"
                    )

    def _yield_attrs() -> Iterator[tuple[str, Any]]:
        for field in xatspec.attributes.values():
            yield (
                field.name,
                self.__dict__.pop(field.name, field.attr.default),
            )

    children = dict(list(_yield_children()))
    attributes = dict(list(_yield_attrs()))

    def _resolve_array(
        field: _Xattribute,
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
                if field.dims is None and (
                    value is None or isinstance(value, _Scalar)
                ):
                    raise CannotExpand(
                        f"Class '{cls_name}' array "
                        f"'{field.name}' can't expand, no dims."
                    )
                if value is None:
                    value = field.attr.default
                if field.dims is None:
                    return value
                shape = tuple([dimensions.pop(dim, dim) for dim in field.dims])
                unresolved = [dim for dim in shape if not isinstance(dim, int)]
                if any(unresolved):
                    if strict:
                        raise DimsNotFound(
                            f"Class '{cls_name}' array "
                            f"'{field.name}' failed dim resolution: "
                            f"{', '.join(unresolved)}"
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
            spec = _xattrs_spec(fields_dict(child_type))
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

    def _yield_arrays() -> Iterator[
        tuple[str, ArrayLike | tuple[str, ArrayLike]]
    ]:
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

    arrays = dict(list(_yield_arrays()))

    setattr(
        self,
        where,
        DataTree(
            Dataset(
                data_vars=arrays,
                coords=coordinates,
                attrs={n: v for n, v in attributes.items()},
            ),
            name=name,
            children={n: getattr(c, where) for n, c in children.items()},
        ),
    )
    _bind_tree(self, parent=parent, children=children)


def _getattribute(self: _HasAttrs, name: str) -> Any:
    cls = type(self)
    if name == (where := cls.__xattree__[_WHERE]):
        raise AttributeError
    if name == cls.__xattree__[_READY]:
        return False

    tree: DataTree = getattr(self, where, None)
    match name:
        case "name":
            return tree.name
        case "dims":
            return tree.dims
        case "parent":
            return None if tree.is_root else tree.parent.self
        case "children":
            # TODO: make `children` a full-fledged attribute?
            return {n: c.self for n, c in tree.children.items()}

    # TODO use xattree spec instead of attrs, and dispatch on
    # the info there instead of introspecting.
    # spec = xattrs_dict(cls)
    spec = fields_dict(cls)
    if field := spec.get(name, None):
        vtype_origin = get_origin(field.type)
        vtype_args = get_args(field.type)
        if (
            vtype_origin
            and isclass(vtype_origin)
            and issubclass(vtype_origin, Iterable)
            and (
                attrs.has(vtype_args[0])
                or (vtype_args[0] is str and attrs.has(vtype_args[1]))
            )
        ):
            if issubclass(vtype_origin, Mapping):
                return {
                    n: c.self
                    for n, c in tree.children.items()
                    if issubclass(type(c.self), vtype_args[1])
                }
            return [
                c.self
                for c in tree.children.values()
                if issubclass(type(c.self), vtype_args[0])
            ]
        value = _get(tree, name, None)
        if isinstance(value, DataTree):
            return value.self
        if value is not None:
            return value
        return None

    raise AttributeError


def _setattribute(self: _HasAttrs, name: str, value: Any):
    cls = type(self)
    cls_name = cls.__name__
    ready = cls.__xattree__[_READY]
    where = cls.__xattree__[_WHERE]

    if not getattr(self, ready, False) or name == ready or name == where:
        self.__dict__[name] = value
        return

    # spec = xattrs_dict(cls)  # TODO see below
    spec = fields_dict(cls)
    if not (field := spec.get(name, None)):
        raise AttributeError(f"{cls_name} has no attribute {name}")

    # TODO use xattree metadata from xattrs_dict to determine
    # how to dispatch the mutation, instead of introspecting.
    # first we need to store the treespec in cls.__xattree__
    match field.type:
        case t if attrs.has(t):
            _bind_tree(
                self,
                children=self.children
                | {field.name: getattr(value, where).self},
            )
        case t if (origin := get_origin(t)) and issubclass(origin, np.ndarray):
            self.data.update({field.name: value})
        case t if not origin:
            self.data.attrs[field.name] = value


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
    metadata[SPEC] = _Dimension(coord=coord, scope=scope)
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
    metadata[SPEC] = _Coordinate(dim=dim, scope=scope)
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
    metadata[SPEC] = _Array(cls=cls, dims=dims)

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


def _xattr(field: attrs.Attribute) -> bool:
    """Check whether `field` is a `xattree` field."""
    return SPEC in field.metadata


def xats(cls) -> bool:
    """Check whether `cls` has cat(-tree attribute)s."""
    if not (meta := getattr(cls, "__xattree__", None)):
        return False
    return meta[_READY]


def fields_dict(cls, xattrs: bool = False) -> dict[str, attrs.Attribute]:
    """
    Get the `attrs` fields of a class. By default, only your
    attributes are returned, not the cat-tree attributes. To
    include cat-tree attributes, set `xattrs=True`.
    """
    return {
        n: f
        for n, f in attrs.fields_dict(cls).items()
        if xattrs or n not in _XATTREE_FIELDS.keys()
    }


def fields(cls, xattrs: bool = False) -> list[attrs.Attribute]:
    """
    Get the `attrs` fields of a class. By default, only your
    attributes are returned, not the cat-tree attributes. To
    include cat-tree attributes, set `xattrs=True`.
    """
    return list(fields_dict(cls, xattrs).values())


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
            setattr(self, cls.__xattree__[_READY], False)

        def post_init(self):
            if orig_post_init:
                orig_post_init(self)
            _init_tree(self, strict=self.strict, where=cls.__xattree__[_WHERE])
            setattr(self, cls.__xattree__[_READY], True)

        def transformer(
            cls: type, fields: list[attrs.Attribute]
        ) -> Iterator[attrs.Attribute]:
            def _transform_field(field):
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
                metadata[SPEC] = _Child(
                    cls=type_,
                    name=field.name,
                    attr=field,
                    kind="dict" if mapping else "list" if iterable else "one",
                )

                default = field.default
                if default is attrs.NOTHING:
                    kwargs = {}
                    if not iterable:
                        kwargs[STRICT] = False
                    default = attrs.Factory(lambda: type_(**kwargs))
                elif default is None and iterable:
                    raise ValueError(
                        "Child collection's default may not be None."
                    )

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

            def _transform(fields_):
                for field in fields_:
                    yield _transform_field(field)

            attrs_ = list(_transform(fields))
            xattrs = [
                f(cls) if isinstance(f, Callable) else f
                for f in _XATTREE_FIELDS.values()
            ]
            return attrs_ + xattrs

        cls.__attrs_pre_init__ = pre_init
        cls.__attrs_post_init__ = post_init
        cls = attrs.define(
            cls,
            slots=False,
            field_transformer=transformer,
        )
        cls.__getattr__ = _getattribute
        cls.__setattr__ = _setattribute
        cls.__xattree__ = {_WHERE: where, _READY: _XATTREE_READY}
        return cls

    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)
