"""
Herd an unruly glaring of `attrs` classes into an orderly `xarray.DataTree`.
"""

import builtins
import json
import types
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import datetime
from enum import Enum
from importlib.metadata import Distribution
from inspect import isclass
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    TypedDict,
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
_Float = float | np.float32 | np.float64
_Numeric = int | float | np.int64 | np.float64
_Scalar = bool | _Numeric | str | Path | datetime
_Array = list | np.ndarray
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


def _drop_none(d: Mapping) -> Mapping:
    return {k: v for k, v in d.items() if v is not None}


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
class _VarSpec:
    cls: Optional[type] = None
    name: Optional[str] = None
    optional: bool = False


@attrs.define
class _DimSpec(_VarSpec):
    coord: Optional["_CoordSpec"] = None
    scope: Optional[str] = None


@attrs.define
class _CoordSpec(_VarSpec):
    dim: Optional[_DimSpec] = None
    scope: Optional[str] = None


@attrs.define
class _ScalarSpec(_VarSpec):
    pass


@attrs.define
class _ArraySpec(_VarSpec):
    dims: Optional[tuple[str, ...]]


@attrs.define
class _ChildSpec(_VarSpec):
    kind: Literal["one", "list", "dict"]


@attrs.define
class _TreeSpec:
    dimensions: dict[str, _DimSpec]
    coordinates: dict[str, _CoordSpec]
    scalars: dict[str, _ScalarSpec]
    arrays: dict[str, _ArraySpec]
    children: dict[str, _ChildSpec]


def _is_child(type_: type) -> bool:
    return type_ and attrs.has(type_)


def _get_var_spec(var: attrs.Attribute) -> _VarSpec:
    """Extract a full variable specification from an `attrs.Attribute`."""

    if var.type is None:
        raise TypeError(f"Field has no type: {var.name}")

    type_ = var.type
    args = get_args(type_)
    origin = get_origin(type_)
    spec = var.metadata.get(SPEC)
    optional = spec.get("optional", False)

    match type(spec):
        case _DimSpec():
            if not (isclass(type_) and issubclass(type, _Int)):
                raise TypeError(f"Dim '{var.name}' must be an integer")
            return attrs.evolve(
                spec,
                name=var.name,
                coord=attrs.evolve(
                    spec.coord,
                    name=var.name if not spec.coord.name else spec.coord.name,
                ),
            )
        case _CoordSpec():
            if not (isclass(origin) and issubclass(origin, _Numeric)):
                raise TypeError(
                    f"Coord field '{var.name}' must be an array type"
                )
            return attrs.evolve(
                spec,
                name=var.name,
                dim=attrs.evolve(
                    spec.dim,
                    name=var.name if not spec.dim.name else spec.dim.name,
                ),
            )
        case _ArraySpec():
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
                        f"Array field must have a concrete type: {var.name}"
                    )
            if not (isclass(origin) and issubclass(origin, _Array)):
                raise TypeError(
                    f"Array field '{var.name}' has unsupported type: {origin}"
                )
            return attrs.evolve(
                spec, name=var.name, cls=type_, optional=optional
            )
        case _ScalarSpec():
            if origin in (Union, types.UnionType):
                if args[-1] is types.NoneType:  # Optional
                    optional = True
                    type_ = args[0]
                else:
                    raise TypeError(
                        f"Scalar field must have a concrete type: {var.name}"
                    )
            if not (isclass(type_) and issubclass(type_, _Scalar)):
                raise TypeError(
                    f"Scalar field '{var.name}' has unsupported type '{type_}'"
                )
            return attrs.evolve(
                spec, name=var.name, cls=type_, optional=optional
            )
        case _ChildSpec():
            if origin:
                if origin not in [list, dict]:
                    raise TypeError(
                        f"Child collection field '{var.name}' "
                        f"must be a list or dictionary"
                    )
                match len(args):
                    case 1:
                        coll = "list"
                        type_ = args[0]
                        if not attrs.has(type_):
                            raise TypeError(
                                f"List field '{var.name}' child "
                                f"type '{type_}' is not attrs"
                            )
                    case 2:
                        coll = "dict"
                        type_ = args[1]
                        if not (args[0] is str and attrs.has(type_)):
                            raise TypeError(
                                f"Dict field '{var.name}' child "
                                f"type '{type_}' is not attrs"
                            )
            else:
                coll = "none"
                if not attrs.has(var.type):
                    raise TypeError(f"Child field '{var.name}' is not attrs")
            return attrs.evolve(
                spec, name=var.name, cls=type_, coll=coll, optional=optional
            )

    raise TypeError(
        f"Field '{var.name}' could not be classified as "
        f"a dim, coord, scalar, array, or child variable"
    )


def _get_tree_spec(attrs_spec: Mapping[str, attrs.Attribute]) -> _TreeSpec:
    """Parse an `attrs` specification into a `xattree` specification."""
    dimensions = {}
    coordinates = {}
    scalars = {}
    arrays = {}
    children = {}

    for var in attrs_spec.values():
        if var.name in _XATTREE_FIELDS.keys():
            continue
        var_spec = _get_var_spec(var)
        match type(var_spec):
            case _DimSpec():
                dimensions[var.name] = var_spec
                coordinates[var_spec[COORD][NAME]] = var_spec[COORD]
            case _CoordSpec():
                dimensions[var_spec[DIM][NAME]] = var_spec[DIM]
                coordinates[var.name] = var_spec
            case _ArraySpec():
                arrays[var.name] = var_spec
            case _ScalarSpec():
                scalars[var.name] = var_spec
            case _ChildSpec():
                children[var.name] = var_spec
            case _:
                raise TypeError(
                    f"Unrecognized var spec type: {type(var_spec)}"
                )

    return _TreeSpec(
        dimensions=dimensions,
        coordinates=coordinates,
        scalars=scalars,
        arrays=arrays,
        children=children,
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
        child_tree = getattr(child, where)
        if n in self.data:
            tree.update({n: child_tree})
        else:
            tree = tree.assign({n: child_tree})
        tree.self = self
        setattr(self, where, tree)
        setattr(child, where, tree[n])
        tree[n].self = child

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
    spec = fields_dict(cls)
    treespec = _get_tree_spec(spec)
    dimensions = {}
    scalars = {}
    arrays = {}

    def _yield_children():
        for var in treespec["children"].values():
            origin = get_origin(var.type)
            if attrs.has(var.type) or (
                origin
                and origin not in (Union, types.UnionType)
                and issubclass(origin, Iterable)
            ):
                if child := self.__dict__.pop(var.name, None):
                    is_iterable = origin and issubclass(origin, Iterable)
                    if is_iterable:
                        if issubclass(origin, Mapping):
                            for k, c in child.items():
                                yield (k, c)
                        else:
                            for i, c in enumerate(child):
                                yield (f"{var.name}_{i}", c)
                    else:
                        yield (var.name, child)

    children = dict(list(_yield_children()))

    def _yield_scalars():
        for var in treespec["scalars"].values():
            yield (var.name, self.__dict__.pop(var.name, var.default))

    scalars = dict(list(_yield_scalars()))

    def _resolve_array(
        attr: attrs.Attribute,
        value: ArrayLike,
        strict: bool = False,
        **dimensions,
    ) -> tuple[Optional[NDArray], Optional[dict[str, NDArray]]]:
        dimensions = dimensions or {}
        dims = attr.metadata.get("dims", None)
        if dims is None and (value is None or isinstance(value, _Scalar)):
            raise CannotExpand(
                f"Class '{cls_name}' array "
                f"'{attr.name}' can't expand, no dims."
            )
        elif value is None:
            value = attr.default
        elif dims is None:
            return value
        shape = tuple([dimensions.pop(dim, dim) for dim in dims])
        unresolved = [dim for dim in shape if not isinstance(dim, int)]
        if any(unresolved):
            if strict:
                raise DimsNotFound(
                    f"Class '{cls_name}' array "
                    f"'{attr.name}' failed dim resolution: "
                    f"{', '.join(unresolved)}"
                )
            return None
        return _chexpand(value, shape)

    def _yield_coords(scope: str) -> Iterator[tuple[str, tuple[str, Any]]]:
        for obj in children.values():
            child_type = type(obj)
            child_origin = get_origin(child_type)
            child_args = get_args(child_type)
            is_iterable = child_origin and issubclass(child_origin, Iterable)
            if is_iterable:
                child_type = child_args[0]
            if not attrs.has(child_type):
                continue
            spec = fields_dict(child_type)
            tree = getattr(obj, where)
            for n, var in spec.items():
                if coord := var.metadata.get("coord", None):
                    if scope == coord.get("scope", None):
                        coord_arr = tree.coords[n].data
                        dimensions[n] = coord_arr.size
                        yield coord.get("dim", n), (n, coord_arr)
                if dim := var.metadata.get("dim", None):
                    if scope == dim.get("scope", None):
                        coord_name = dim.get("coord", n)
                        coord_arr = tree.coords[coord_name].data
                        dimensions[n] = coord_arr.size
                        yield coord_name, (n, coord_arr)
        if parent:
            parent_tree = getattr(parent, where)
            for coord_name, coord in parent_tree.coords.items():
                dim_name = coord.dims[0]
                dimensions[dim_name] = coord.data.size
                yield (coord_name, (dim_name, coord.data))
        for var in treespec["arrays"].values():
            if not (coord := var.metadata.get("coord", None)):
                continue
            if (
                array := _resolve_array(
                    var,
                    value=self.__dict__.pop(var.name, var.default),
                    strict=strict,
                )
            ) is not None:
                dim_name = coord.get("dim", var.name)
                dimensions[dim_name] = array.size
                yield (var.name, (dim_name, array))
        for scalar_name, scalar in scalars.items():
            if scalar_name not in treespec["dimensions"]:
                continue
            coord = treespec["coordinates"][scalar_name]
            match type(scalar):
                # TODO is splitting out int/float cases necessary?
                case builtins.int | np.int64:
                    step = coord.get("step", 1)
                    start = 0
                case builtins.float | np.float64:
                    step = coord.get("step", 1.0)
                    start = 0.0
                case _:
                    raise ValueError("Dimensions/coordinates must be numeric.")
            coord_arr = np.arange(start, scalar, step)
            dimensions[scalar_name] = coord_arr.size
            yield (
                coord.get("name", scalar_name),
                (scalar_name, coord_arr),
            )

    coordinates = dict(list(_yield_coords(scope=cls_name)))

    def _yield_arrays():
        explicit_dims = self.__dict__.pop("dims", None) or {}
        for var in treespec["arrays"].values():
            dims = var.metadata.get("dims", None)
            if var.metadata.get("coord", False):
                continue
            if (
                array := _resolve_array(
                    var,
                    value=self.__dict__.pop(var.name, var.default),
                    strict=strict,
                    **dimensions | explicit_dims,
                )
            ) is not None and var.default is not None:
                yield (var.name, (dims, array) if dims else array)

    arrays = dict(list(_yield_arrays()))

    setattr(
        self,
        where,
        DataTree(
            Dataset(
                data_vars=arrays,
                coords=coordinates,
                attrs={n: v for n, v in scalars.items()},
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
    if var := spec.get(name, None):
        vtype_origin = get_origin(var.type)
        vtype_args = get_args(var.type)
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
    if not (attr := spec.get(name, None)):
        raise AttributeError(f"{cls_name} has no attribute {name}")

    # TODO use xattree metadata from xattrs_dict to determine
    # how to dispatch the mutation, instead of introspecting.
    # first we need to store the treespec in cls.__xattree__
    match attr.type:
        case t if attrs.has(t):
            _bind_tree(
                self,
                children=self.children
                | {attr.name: getattr(value, where).self},
            )
        case t if (origin := get_origin(t)) and issubclass(origin, _Array):
            self.data.update({attr.name: value})
        case t if not origin and issubclass(attr.type, _Scalar):
            self.data.attrs[attr.name] = value


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
    metadata[SPEC] = _DimSpec(coord=coord, scope=scope)
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
    metadata[SPEC] = _CoordSpec(dim=dim, scope=scope)
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

    dims = dims if isinstance(dims, Iterable) else None
    if not any(dims) and isinstance(default, _Scalar):
        raise CannotExpand("If no dims, no scalar defaults.")

    if cls and default is attrs.NOTHING:
        default = attrs.Factory(cls)

    metadata = metadata or {}
    metadata[SPEC] = _ArraySpec(cls=cls, dims=dims)

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


def child(
    cls,
    default=attrs.NOTHING,
    validator=None,
    repr=True,
    eq=True,
    metadata=None,
):
    """
    Create a child field. The child type must be an `attrs`-based class.
    """

    origin = get_origin(cls)
    iterable = origin and issubclass(origin, Iterable)
    mapping = origin and issubclass(origin, Mapping)
    if default is attrs.NOTHING:
        kwargs = {}
        if not iterable:
            kwargs[STRICT] = False
        default = attrs.Factory(lambda: cls(**kwargs))
    elif default is None and iterable:
        raise ValueError("Child collection's default may not be None.")

    metadata = metadata or {}
    metadata[SPEC] = _ChildSpec(
        cls=cls, kind="dict" if mapping else "list" if iterable else "one"
    )

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


@dataclass_transform(field_specifiers=(attrs.field, dim, coord, array, child))
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
        ) -> list[attrs.Attribute]:
            return fields + [
                f(cls) if isinstance(f, Callable) else f
                for f in _XATTREE_FIELDS.values()
            ]

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
