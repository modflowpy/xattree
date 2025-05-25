"""
Herd an unruly glaring of `attrs` classes into an orderly `xarray.DataTree`.
"""

import builtins
import types
from collections import ChainMap
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, MutableSequence
from datetime import datetime
from inspect import isclass
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    overload,
)

import numpy as np
import xarray as xr
from attrs import NOTHING, Attribute, Converter, Factory, cmp_using, define, evolve
from attrs import (
    field as attrs_field,
)
from attrs import (
    fields_dict as attrs_fields_dict,
)
from attrs import (
    has as attrs_has,
)
from numpy.typing import ArrayLike, NDArray
from xarray.core.indexes import PandasIndex

_PKG_NAME = "xattree"


class DataTreeList(MutableSequence):
    """Proxy a `DataTree`'s children of a given type through a list-like interface."""

    def __init__(self, tree: xr.DataTree, type_: type, where: str, prefix: str):
        self._tree = tree
        self._type = type_
        self._where = where
        self._prefix = prefix
        self._cache = self._build_cache()

    def _build_cache(self) -> list[Any]:
        return [
            c.attrs[_HOST]
            for c in self._tree.children.values()
            if issubclass(type(c.attrs[_HOST]), self._type)  # type: ignore
        ]

    def __eq__(self, value):
        return self._cache == value

    def __len__(self) -> int:
        return len(self._cache)

    @overload
    def __getitem__(self, index: int) -> Any: ...

    @overload
    def __getitem__(self, index: slice) -> MutableSequence[Any]: ...

    def __getitem__(self, index: int | slice) -> Any | MutableSequence[Any]:
        return self._cache[index]

    @overload
    def __setitem__(self, index: int, value: Any) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Any]) -> None: ...

    def __setitem__(self, index: int | slice, value: Any | Iterable[Any]) -> None:
        def _set(host, key, val):
            new_node = getattr(val, self._where)
            new_children = dict(self._tree.children) | {key: new_node}
            self._tree = self._tree.assign(new_children)
            self._tree.attrs[_HOST] = host
            setattr(host, self._where, self._tree)

        host = self._tree.attrs[_HOST]
        if isinstance(index, slice):
            for i, v in enumerate(value):
                key = f"{self._prefix}{index.start + i}"
                _set(host, key, v)
        else:
            key = f"{self._prefix}{index}"
            _set(host, key, value)

        self._cache = self._build_cache()

    @overload
    def __delitem__(self, index: int) -> None: ...

    @overload
    def __delitem__(self, index: slice) -> None: ...

    def __delitem__(self, index: int | slice) -> None:
        if isinstance(index, slice):
            for i in range(index.start or 0, index.stop or len(self._cache)):
                key = f"{self._prefix}{i}"
                del self._tree[key]
        else:
            key = f"{self._prefix}{index}"
            del self._tree[key]
        self._cache = self._build_cache()

    def __iter__(self):
        return iter(self._cache)

    def __repr__(self):
        return list.__repr__(self._cache)

    def insert(self, index: int, value: Any):
        self.__setitem__(index, value)


class DataTreeDict(MutableMapping):
    """Proxy a `DataTree`'s children of a given type through a dict-like interface."""

    def __init__(self, tree: xr.DataTree, type_: type, where: str):
        self._tree = tree
        self._type = type_
        self._where = where
        self._cache = self._build_cache()

    def _build_cache(self) -> dict[str, Any]:
        return {
            n: c.attrs[_HOST]
            for n, c in self._tree.children.items()
            if issubclass(type(c.attrs[_HOST]), self._type)  # type: ignore
        }

    def __eq__(self, value):
        return self._cache == value

    def __or__(self, other):
        return dict(self._cache) | dict(other)

    def __ior__(self, _):
        raise NotImplementedError("In-place merge is not supported")

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, key: str) -> Any:
        return self._cache[key]

    def __setitem__(self, key: str, value: Any):
        host = self._tree.attrs[_HOST]
        new_node = getattr(value, self._where)
        new_children = dict(self._tree.children) | {key: new_node}
        self._tree = self._tree.assign(new_children)
        self._tree.attrs[_HOST] = host
        setattr(host, self._where, self._tree)
        self._build_cache()

    def __delitem__(self, key: str):
        del self._tree[key]
        self._build_cache()

    def __iter__(self):
        return iter(self._cache)

    def __repr__(self):
        return dict.__repr__(self._cache)


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


class ROOT:
    """Lift the scope of a dimension or coordinate to the root of the tree."""

    pass


Int = int | np.integer
Float = float | np.floating
Numeric = Int | Float
Scalar = bool | Numeric | str | Path | datetime
_NAME = "name"
_DATA = "data"
_HOST = "host"
_KIND = "kind"
_COORD = "coord"
_DIMS = "dims"
_SCOPE = "scope"
_SPEC = "spec"
_STRICT = "strict"
_TYPE = "type"
_OPTIONAL = "optional"
_CONVERTER = "converter"
_CONVERTERS = "converters"
_VALIDATOR = "validator"
_VALIDATORS = "validators"
_PARENT = "parent"
_CHILDREN = "children"
_MULTI = "multi"
_CLASS = "class"
_INDEX = "index"
_INDEX_SCOPE = f"{_INDEX}_{_SCOPE}"
_WHERE = "where"
_XATTREE_DUNDER = "__xattree__"
_XATTREE_READY = "_xattree_ready"
_XTRA_ATTRS = {
    _NAME: lambda cls: Attribute(  # type: ignore
        name=_NAME,
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
    _DATA: lambda cls: Attribute(  # type: ignore
        name=getattr(cls, _XATTREE_DUNDER, {}).get(_WHERE, _DATA),
        default=None,
        validator=None,
        repr=False,
        cmp=None,
        hash=False,
        eq=False,
        init=False,
        inherited=False,
        type=xr.DataTree,
    ),
    _DIMS: Attribute(  # type: ignore
        name=_DIMS,
        default=Factory(dict),
        validator=None,
        repr=False,
        cmp=None,
        hash=False,
        eq=False,
        init=True,
        inherited=False,
        type=Mapping[str, int],
    ),
    _PARENT: Attribute(  # type: ignore
        name=_PARENT,
        default=None,
        validator=None,
        repr=False,
        cmp=None,
        hash=False,
        eq=False,
        init=True,
        inherited=False,
        type=Any,
    ),
    _CHILDREN: Attribute(  # type: ignore
        name=_CHILDREN,
        default=Factory(dict),
        validator=None,
        repr=False,
        cmp=None,
        hash=False,
        eq=False,
        init=False,
        inherited=False,
        type=Mapping[str, Any],
    ),
    _STRICT: Attribute(  # type: ignore
        name=_STRICT,
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
_XTRA_GETTERS = {
    _NAME: lambda tree: tree.name,
    _DIMS: lambda tree: tree.dims,
    _PARENT: lambda tree: None if tree.is_root else tree.parent.attrs[_HOST],
    _CHILDREN: lambda tree: DataTreeDict(tree, type_=object, where=_DATA),
    _STRICT: lambda _: False,
}
_XTRA_SETTERS = {
    _NAME: lambda tree, _, value: setattr(tree, _NAME, value),
}


def chexpand(value: ArrayLike, shape: tuple[int]) -> NDArray:
    """
    CHeck an array-like value's shape. If it's a scalar, EXPAND it
    to the requested shape. If an array, make sure it's that shape.
    """

    try:
        shp = value.shape  # type: ignore
    except AttributeError:
        return np.full(shape, value)
    except Exception:
        raise ValueError(f"Unsupported array item type : {type(value)}")
    if shp == ():
        return np.full(shape, value.item())  # type: ignore
    if shp != shape:
        raise ValueError(f"Shape mismatch, got {shp}, expected {shape}")
    # any way to avoid the cast?
    return cast(NDArray, value)


@define
class Xattribute:
    """Specifies a `xattree`-decorated field."""

    name: str
    default: Optional[Any] = None
    optional: bool = False
    type: Optional["type"] = None
    converter: Optional[Callable] = None
    metadata: Optional[dict[str, Any]] = None


@define
class Attr(Xattribute):
    """Specifies a field that is not a dimension, coordinate, or array."""

    pass


@define
class Array(Xattribute):
    """Specifies an array field."""

    dims: Optional[tuple[str, ...]] = None
    dtype: Optional["type"] = None


@define
class Coord(Xattribute):
    """Specifies a coordinate field."""

    path: Optional[str] = None
    scope: Optional[str] = None
    dim: Optional[str] = None


@define
class Dim(Xattribute):
    """Specifies a dimension field."""

    path: Optional[str] = None
    scope: Optional[str] = None
    coord: Optional[bool | str] = True


ChildKind = Literal["only", "list", "dict"]
"""Specifies the kind of child field."""


@define
class Child(Xattribute):
    """Specifies a child field, i.e. another node in the tree."""

    type: Optional["type"] = None
    kind: ChildKind = "only"


@define
class XatSpec:
    """Specifies a `xattree`-decorated class."""

    dims: dict[str, Dim]
    attrs: dict[str, Attr]
    arrays: dict[str, Array]
    coords: dict[str, Coord]
    children: dict[str, Child]

    @property
    def flat(self) -> MutableMapping[str, Xattribute]:
        return ChainMap(self.dims, self.attrs, self.arrays, self.coords, self.children)  # type: ignore


def _get_xatspec(cls: type) -> XatSpec:
    """Extract a `xattree` specification from a given class."""
    cls_name = cls.__name__

    def __get_xatspec(fields: dict) -> XatSpec:
        dims = {}
        attributes = {}
        arrays = {}
        coords = {}
        children = {}

        def _register_nested_dims(child_spec: Child, path=None):
            if child_spec.type is None:
                return
            for child in (spec := _get_xatspec(child_spec.type)).children.values():
                if child.type:
                    _register_nested_dims(
                        child, path=f"{path}/{child.name}" if path else child.name
                    )
            cls_name_l = cls_name.lower()
            for dim_name, dim in spec.dims.items():
                if dim.scope is ROOT or dim.scope == cls_name_l:
                    dims[dim_name] = evolve(
                        dim,
                        path=f"{path}/{child_spec.name}" if path else child_spec.name,
                    )
            for coord_name, coord in spec.coords.items():
                if coord.scope is ROOT or coord.scope == cls_name_l:
                    coords[coord_name] = evolve(
                        coord,
                        path=f"{path}/{child_spec.name}" if path else child_spec.name,
                    )

        for field in fields.values():
            if field.name in _XTRA_ATTRS.keys():
                continue
            if field.type is None:
                raise TypeError(f"Field has no type: {field.name}")
            type_ = field.type
            args = get_args(type_)
            origin = get_origin(type_)
            metadata = field.metadata.copy()
            if (xatmeta := metadata.pop(_PKG_NAME, None)) is None:
                continue
            is_optional = xatmeta.get(_OPTIONAL, False)
            match xatmeta.get(_KIND, None):
                case "dim":
                    if origin in (Union, types.UnionType):
                        if args[-1] is types.NoneType:  # Optional
                            is_optional = True
                            type_ = args[0]
                        else:
                            raise TypeError(f"Dim must have a concrete type: {field.name}")
                    if not (isclass(type_) and issubclass(type_, Int)):
                        raise TypeError(f"Dim '{field.name}' must be an integer")
                    dims[field.name] = Dim(
                        name=field.name,
                        default=field.default,
                        optional=is_optional,
                        metadata=metadata,
                        coord=xatmeta.get(_COORD, False),
                        scope=xatmeta.get(_SCOPE, None),
                        type=type_,
                    )
                case "coord":
                    if not (isclass(origin) and issubclass(origin, np.ndarray)):
                        raise TypeError(f"Coord '{field.name}' must be an array type")
                    coords[field.name] = Coord(
                        name=field.name,
                        default=field.default,
                        optional=is_optional,
                        metadata=metadata,
                        scope=xatmeta.get(_SCOPE, None),
                        type=type_,
                    )
                case "array":
                    dtype = None
                    if origin in (Union, types.UnionType):
                        if args[-1] is types.NoneType:  # Optional
                            is_optional = True
                            type_ = args[0]
                            if get_origin(type_) is np.ndarray:
                                origin = np.ndarray
                                dtype = get_args(type_)[1].__args__[0]
                            elif get_origin(type_) is list:
                                origin = list
                                dtype = get_args(type_)[0]
                            else:
                                origin = None
                        else:
                            raise TypeError(f"Field must have a concrete type: {field.name}")
                    if not (isclass(origin) and issubclass(origin, (list, np.ndarray))):
                        raise TypeError(f"Array '{field.name}' type unsupported: {origin}")
                    arrays[field.name] = Array(
                        dims=xatmeta[_DIMS],
                        name=field.name,
                        default=field.default,
                        optional=is_optional,
                        type=type_,
                        dtype=dtype,
                        converter=field.converter,
                        metadata=metadata,
                    )
                case "child" | "attr" | None:
                    child_kind: ChildKind | None = None
                    is_child = False
                    is_optional = False
                    iterable = isclass(origin) and issubclass(origin, Iterable)
                    mapping = iterable and issubclass(origin, Mapping)
                    if origin in (Union, types.UnionType):
                        if args[-1] is types.NoneType:  # Optional
                            is_optional = True
                            origin = None
                            type_ = args[0]
                    elif not origin and has(type_):
                        is_child = True
                        child_kind = "only"
                    elif iterable or mapping:
                        match len(args):
                            case 1:
                                type_ = args[0]
                                if has(type_):
                                    is_child = True
                                    child_kind = "list"
                            case 2:
                                type_ = args[1]
                                if args[0] is str and has(type_):
                                    is_child = True
                                    child_kind = "dict"
                    if is_child:
                        child = Child(
                            type=type_,
                            name=field.name,
                            default=field.default,
                            optional=is_optional,
                            kind=child_kind or "only",
                            metadata=metadata,
                        )
                        children[field.name] = child
                        _register_nested_dims(child)
                    else:
                        attributes[field.name] = Attr(
                            name=field.name,
                            type=field.type,
                            default=field.default,
                            optional=is_optional,
                            metadata=metadata,
                        )

        return XatSpec(dims=dims, attrs=attributes, arrays=arrays, coords=coords, children=children)

    if (meta := getattr(cls, _XATTREE_DUNDER, None)) and meta[_CLASS] == cls:
        return meta[_SPEC]

    return __get_xatspec(fields_dict(cls))


def get_xatspec(cls: type) -> Mapping[str, Xattribute]:
    """
    Get the `xattree` specification for a given class.

    Parameters
    ----------
    cls : type
        The class to get the specification for.

    Returns
    -------
    Mapping
        The `xattree` specification for the class.

    Raises
    ------
    TypeError
        If the class is not decorated with `xattree`.
    """
    if not getattr(cls, _XATTREE_DUNDER, None):
        raise TypeError(f"Class '{cls.__name__}' is not decorated with xattree.")

    return _get_xatspec(cls).flat


def _bind_tree(
    self: Any,
    parent: Any = None,
    children: Optional[Mapping[str, Any]] = None,
    where: str = _DATA,
):
    """
    Bind a tree to its parent and children, and give each tree node
    a reference to its host.
    """
    name = getattr(self, where).name
    tree = getattr(self, where)
    children = children or {}
    cls = type(self)

    # bind parent
    if parent:
        parent_cls = type(parent)
        parent_spec = get_xatspec(parent_cls)

        def _find_field(cls: type) -> str:
            matches = set()
            for name, field in parent_spec.items():
                if isinstance(field, Child) and isclass(field.type) and issubclass(cls, field.type):
                    matches.add(name)
            match len(matches):
                case 0:
                    raise TypeError(
                        f"Class '{parent_cls.__name__}' has no fields of type {cls.__name__}"
                    )
                case 1:
                    return matches.pop()
                case _:
                    raise TypeError(
                        f"Class '{parent_cls.__name__}' has multiple fields of type "
                        f"{cls.__name__}' ({', '.join(matches)}), can't bind."
                    )

        parent_field = _find_field(cls)
        if (field := parent_spec.get(parent_field, None)) is None:
            raise TypeError(f"Class '{parent_cls.__name__}' has no field '{parent_field}'")
        if not isinstance(field, Child):
            raise TypeError(f"Class '{parent_cls.__name__}' field '{parent_field}' is not a child")

        parent_tree = getattr(parent, where)
        siblings = {n: c for n, c in parent_tree.children.items()}

        def _update_or_assign(field: Child, name: str) -> tuple[str, bool, dict]:
            match field.kind:
                case "only":
                    if name in parent.data:
                        return name, True, {name: tree}
                    else:
                        return name, False, {name: tree, **siblings}
                case "list":
                    same_type = {n: c for n, c in siblings.items() if type(c.attrs[_HOST]) is cls}
                    name = f"{name}{len(same_type)}"
                    return name, name in parent.data, siblings | {name: tree}
                case "dict":
                    return name, name in parent.data, siblings | {name: tree}

        name, update, new_siblings = _update_or_assign(field, name)
        if update:
            parent_tree.update(new_siblings)
            parent_tree.attrs[_HOST] = parent
            setattr(parent, where, parent_tree)
            tree = parent_tree[name]
            setattr(self, where, tree)
        else:
            is_root = parent_tree.is_root
            lineage = parent_tree.parents
            parent_tree = parent_tree.assign(new_siblings)
            parent_tree.attrs[_HOST] = parent
            setattr(parent, where, parent_tree)
            _bind_tree(
                parent,
                children={n: s.attrs[_HOST] for n, s in new_siblings.items()},
            )
            tree = parent_tree[name]
            setattr(self, where, tree)
            if not is_root:
                other = {parent_tree.name: parent_tree}
                for ancestor in lineage:
                    ancestor.update(other)
                    if not ancestor.is_root:
                        other = {ancestor.name: ancestor}
                parent_tree = lineage[0][parent_tree.name]
                setattr(parent, where, parent_tree)

    # bind children
    for n, sibling in children.items():
        child_tree = getattr(sibling, where)
        tree[n].attrs[_HOST] = sibling
        setattr(sibling, where, tree[n])
        _bind_tree(
            sibling,
            children={n: c.attrs[_HOST] for n, c in child_tree.children.items()},
            where=where,
        )

    tree.attrs[_HOST] = self
    setattr(self, where, tree)


def _init_tree(
    self: Any,
    strict: bool = True,
    where: str = _DATA,
    index: Callable[[xr.Dataset], xr.Index] | None = None,
) -> None:
    """
    Initialize a `DataTree` for an instance of a `xattree`-decorated class.

    Notes
    -----
    This function must run after the default `__init__()`.

    The tree is built from the class' `attrs` fields, i.e.
    spirited from the instance's `__dict__` into the tree,
    which is added as an attribute named by value `where`.
    `__dict__` is emptyish after this method runs (except
    the data tree and a few other things). Field access is
    proxied to the tree.

    The decorated class cannot use slots for this to work.
    """
    cls = type(self)
    cls_name = cls.__name__
    name = self.__dict__.pop(_NAME, cls_name.lower())
    parent = self.__dict__.pop(_PARENT, None)
    explicit_dims = self.__dict__.pop(_DIMS, None) or {}
    xatspec = _get_xatspec(cls)

    def _yield_children() -> Iterator[tuple[str, Any]]:
        for child in self.__dict__.pop(_CHILDREN, {}).values():
            yield child
        for xat in xatspec.children.values():
            if (child := self.__dict__.pop(xat.name, None)) is None:
                continue
            match xat.kind:
                case "only":
                    yield (xat.name, child)
                case "list":
                    for i, c in enumerate(child):
                        yield (f"{xat.name}{i}", c)
                case "dict":
                    for k, c in child.items():
                        yield (k, c)
                case _:
                    raise TypeError(f"Bad child collection field '{xat.name}'")

    def _yield_attrs() -> Iterator[tuple[str, Any]]:
        yield (_HOST, self)
        for xat_name, xat in chain(xatspec.dims.items(), xatspec.attrs.items()):
            if isinstance(xat, Dim) and xat.coord:
                continue
            yield (xat_name, self.__dict__.pop(xat_name, explicit_dims.get(xat_name, xat.default)))

    children = dict(list(_yield_children()))
    attributes = dict(list(_yield_attrs()))

    def _resolve_array(
        xat: Xattribute, value: ArrayLike, strict: bool = False, **dims
    ) -> Optional[NDArray]:
        dims = dims or {}
        match xat:
            case Coord():
                if xat.default is None or not isinstance(xat.default, Scalar):
                    raise CannotExpand(
                        f"Class '{cls_name}' coord array '{xat.name}'"
                        f"paired with dim '{xat.name}' can't expand "
                        f"without a scalar default dimension size."
                    )
                return chexpand(value, (xat.default,))
            case Array():
                shape = tuple([dims.pop(dim, dim) for dim in (xat.dims or [])])
                unresolved = [dim for dim in shape if not isinstance(dim, int)]
                if strict and any(unresolved):
                    raise DimsNotFound(
                        f"Class '{cls_name}' array '{xat.name}' "
                        f"failed dim resolution: {', '.join(unresolved)}"
                    )
                if value is None or isinstance(value, str) or not isinstance(value, Iterable):
                    if xat.dims is None:
                        raise CannotExpand(
                            f"Class '{cls_name}' array '{xat.name}' can't expand "
                            "without explicit dimensions or a non-scalar default."
                        )
                    value = value if value is not None else xat.default  # type: ignore
                    if value is None:
                        return None  # type: ignore
                    return None if any(unresolved) else chexpand(value, shape)
                value = np.array(value)
                if xat.dims and value.ndim != len(shape):
                    raise ValueError(
                        f"Class '{cls_name}' array '{xat.name}' "
                        f"expected {len(shape)} dims, got {value.ndim}"
                    )
                return value
        return None

    def _find_dim_or_coord(
        children: Mapping[str, Any],
        dim_or_coord: Xattribute,
    ) -> Optional[Union[ArrayLike, Scalar]]:
        match dim_or_coord:
            case Dim() as dim:
                if not dim.path:
                    return None
                dim_name = dim.name
                child_name, _, path = dim.path.partition("/")
                match len(path):
                    case 1:
                        if (child := children.get(child_name, None)) is None:
                            return None
                        child_node = getattr(child, where)
                        return child_node.dims[dim_name]
                    case _:
                        if (child := children.get(child_name, None)) is None:
                            return None
                        child_node = getattr(child, where)
                        target_node = child_node[path]
                        try:
                            return target_node.dims[dim_name]
                        except KeyError:
                            raise KeyError(
                                f"Dim '{dim_name}' declared but not found in "
                                f"scope '{child_name}', is it initialized? If a "
                                f"derived dim/coord, make sure you're using the "
                                f"__attrs_post_init__() method to initialize it."
                            )
            case Coord() as coord:
                if not coord.path:
                    return None
                coord_name = coord.name
                child_name, _, path = coord.path.partition("/")
                match len(path):
                    case 1:
                        if (child := children.get(child_name, None)) is None:
                            return None
                        child_node = getattr(child, where)
                        if coord.dim:
                            return child_node.dims[coord_name]
                        return child_node.coords[coord_name].data
                    case _:
                        if (child := children.get(child_name, None)) is None:
                            return None
                        child_node = getattr(child, where)
                        target_node = child_node[path]
                        try:
                            return (
                                target_node.dims[coord_name]
                                if coord.dim
                                else target_node.coords[coord_name].data
                            )
                        except KeyError:
                            raise KeyError(
                                f"Coord '{coord_name}' declared but not found in "
                                f"scope '{child_name}', is it initialized? If a "
                                f"derived dim/coord, make sure you're using the "
                                f"__attrs_post_init__() method to initialize it."
                            )

        return None

    dimensions = {}
    aliased_coords = []

    def _yield_coords() -> Iterator[tuple[str, tuple[str, NDArray]]]:
        # register inherited dimension sizes so we can expand arrays
        if parent:
            parent_tree: xr.DataTree = getattr(parent, where)
            for dim_name, dim in parent_tree.dims.items():
                dimensions[dim_name] = dim
            for coord in parent_tree.coords.values():
                dimensions[coord.dims[0]] = coord.data.size

        # yield coord arrays, expanding from dim sizes if necessary
        known_dims = dimensions | explicit_dims
        for field_name, dim_or_coord in chain(xatspec.coords.items(), xatspec.dims.items()):
            value = self.__dict__.pop(dim_or_coord.name, None)
            if value is None or value is NOTHING:
                value = known_dims.get(field_name, None) or _find_dim_or_coord(
                    children, dim_or_coord
                )
            if value is None or value is NOTHING:
                value = attributes.get(dim_or_coord.name, None)
            if value is None or value is NOTHING:
                value = dim_or_coord.default
            if value is None or value is NOTHING:
                value = attributes.get(field_name, None)
            if value is None:
                continue
            if isinstance(dim_or_coord, Dim) and not dim_or_coord.coord:
                dimensions[field_name] = value
                attributes[field_name] = value
                continue
            if isinstance(value, Scalar):
                match type(value):
                    case builtins.int | builtins.float | np.number:
                        # todo customizable step/start?
                        step = 1
                        start = 0
                    case _:
                        raise ValueError("Dim size must be numeric.")
                array: np.ndarray = np.arange(start, value, step)
            else:
                array = np.array(value)
            dimensions[field_name] = len(array)
            coord_name = field_name
            if isinstance(dim_or_coord, Dim) and isinstance(dim_or_coord.coord, str):
                coord_name = dim_or_coord.coord
                aliased_coords.append(coord_name)
            yield (coord_name, (field_name, array))

    # resolve dimensions/coordinates before arrays
    coordinates = dict(list(_yield_coords()))

    def _yield_arrays() -> Iterator[tuple[str, NDArray | tuple[tuple[str, ...], NDArray]]]:
        for xat in xatspec.arrays.values():
            if (
                array := _resolve_array(
                    xat,
                    value=self.__dict__.pop(xat.name, xat.default),
                    strict=strict,
                    **dimensions | explicit_dims,
                )
            ) is not None:
                if xat.dims:
                    yield (xat.name, (xat.dims, array))
                else:
                    yield (xat.name, array)

    arrays = dict(list(_yield_arrays()))
    dataset = xr.Dataset(
        data_vars=arrays,
        coords=coordinates,
        attrs={n: a for n, a in attributes.items()},
    )

    def _find_index(children: Mapping[str, Any]) -> Optional[Callable[[xr.Dataset], xr.Index]]:
        for child in children.values():
            child_cls = type(child)
            index = child_cls.__xattree__[_INDEX]
            scope = child_cls.__xattree__[_INDEX_SCOPE]
            cls_name_l = cls.__name__.lower()
            if index and (scope == ROOT or scope == cls_name_l):
                return index
            if (child_index := _find_index(child.children)) is not None:
                return child_index
        return None

    if index := index or _find_index(children):
        dataset = dataset.assign_coords(xr.Coordinates.from_xindex(index(dataset)))

    for ac in aliased_coords:
        dataset = dataset.set_xindex(ac, PandasIndex)

    setattr(
        self,
        where,
        xr.DataTree(
            dataset=dataset,
            name=name,
            children={n: getattr(c, where) for n, c in children.items()},
        ),
    )
    _bind_tree(self, parent=parent, children=children)


def _getattr(self: Any, name: str) -> Any:
    cls = type(self)
    if name == (where := cls.__xattree__[_WHERE]):
        raise AttributeError
    if name == _XATTREE_READY:
        return False
    tree = cast(xr.DataTree, getattr(self, where, None))
    if get_xattr := _XTRA_GETTERS.get(name, None):
        return get_xattr(tree)
    spec = _get_xatspec(cls)
    if xat := spec.flat.get(name, None):
        match xat:
            case Dim():
                try:
                    return tree.dims[xat.name]
                except KeyError:
                    return tree.attrs[name]
            case Coord():
                if xat.dim:
                    try:
                        return tree.dims[xat.name]
                    except KeyError:
                        return tree.attrs[name]
                return tree.coords[xat.name].data
            case Attr():
                return tree.attrs[xat.name]
            case Array():
                try:
                    return tree[xat.name]
                except KeyError:
                    return None
            case Child():
                match xat.kind:
                    case "dict":
                        return DataTreeDict(tree, type_=xat.type, where=where)  # type: ignore
                    case "list":
                        return DataTreeList(tree, type_=xat.type, where=where, prefix=xat.name)  # type: ignore
                    case "only":
                        if (child := tree.children.get(xat.name, None)) is not None:
                            return child.attrs[_HOST]
                        return None
            case _:
                raise TypeError(
                    f"Field '{name}' is not a dimension, coordinate, "
                    "attribute, array, or child variable"
                )

    return super(type(self), self).__getattribute__(name)


def field(
    default=NOTHING,
    validator=None,
    converter=None,
    repr=True,
    eq=True,
    init=True,
    metadata=None,
):
    """Create a field."""
    metadata = metadata or {}
    metadata[_PKG_NAME] = {
        _KIND: None,  # infer later
        _CONVERTER: converter,
        _VALIDATOR: validator,
    }
    return attrs_field(
        default=default,
        repr=repr,
        eq=eq,
        order=False,
        hash=True,
        init=init,
        metadata=metadata,
    )


def dim(
    scope=None,
    coord: bool | str = True,
    default=NOTHING,
    repr=True,
    eq=True,
    init=True,
    metadata=None,
):
    """Create a dimension field."""
    metadata = metadata or {}
    metadata[_PKG_NAME] = {
        _KIND: "dim",
        _COORD: coord,
        _SCOPE: scope,
    }
    return attrs_field(
        default=default,
        repr=repr,
        eq=eq,
        order=False,
        hash=True,
        init=init,
        metadata=metadata,
    )


def coord(
    scope=None,
    default=NOTHING,
    repr=True,
    eq=True,
    metadata=None,
):
    """Create a coordinate field."""
    metadata = metadata or {}
    metadata[_PKG_NAME] = {
        _KIND: "coord",
        _SCOPE: scope,
    }
    return attrs_field(
        default=default,
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
    default=NOTHING,
    validator=None,
    converter=None,
    repr=True,
    eq=None,
    metadata=None,
):
    """Create an array field."""
    dims = dims if isinstance(dims, Iterable) else tuple()
    if not any(dims) and isinstance(default, Scalar):
        raise CannotExpand("If no dims, no scalar defaults.")
    if cls and default is NOTHING:
        default = Factory(cls)
    metadata = metadata or {}
    metadata[_PKG_NAME] = {
        _KIND: "array",
        _DIMS: dims,
        _TYPE: cls,
        _CONVERTER: converter,
        _VALIDATOR: validator,
    }
    return attrs_field(
        default=default,
        repr=repr,
        eq=eq or cmp_using(eq=np.array_equal),
        order=False,
        hash=False,
        init=True,
        metadata=metadata,
    )


def is_xat(field: Attribute) -> bool:
    """Check whether `field` is a `xattree` attribute."""
    return _PKG_NAME in field.metadata


def has(cls) -> bool:
    """Check whether `cls` is a `xattree`."""
    return hasattr(cls, _XATTREE_DUNDER)


def fields_dict(cls, extra: bool = False) -> dict[str, Attribute]:
    """
    Get the field dict for a class. By default, only your
    attributes are included, none of the extra attributes
    set up by `xattree`. To include those set `extra=True`.
    """
    return {n: f for n, f in attrs_fields_dict(cls).items() if extra or n not in _XTRA_ATTRS.keys()}


def fields(cls, extra: bool = False) -> list[Attribute]:
    """
    Get the field list for a class. By default, only your
    attributes are included, none of the extra attributes
    set up by `xattree`. To include those set `extra=True`.
    """
    return list(fields_dict(cls, extra=extra).values())


T = TypeVar("T")


@overload
def xattree(
    *,
    where: str = _DATA,
    index: Callable[[xr.Dataset], xr.Index] | None = None,
    index_scope: str | type | None = None,
) -> Callable[[type[T]], type[T]]: ...


@overload
def xattree(maybe_cls: type[T]) -> type[T]: ...


@dataclass_transform(field_specifiers=(attrs_field, field, dim, coord, array))
def xattree(
    maybe_cls: Optional[type[Any]] = None,
    *,
    where: str = _DATA,
    index: Callable[[xr.Dataset], xr.Index] | None = None,
    index_scope: str | type | None = None,
) -> type[T] | Callable[[type[T]], type[T]]:
    """
    Make an `attrs`-based class a (node in a) `xattree`.

    Parameters
    ----------
    maybe_cls : type, optional
        The class to be decorated. If not provided, the decorator
        is returned as a callable that can be used to decorate
        a class later.
    where : str, optional
        The name of the attribute that will hold the `xattree`.
        Default is "data".
    index : Callable, optional
        A function that takes a `xarray.Dataset` and returns
        an `xarray.Index`. If provided, the index built will
        be assigned as coordinates to the dataset.
    index_scope : str or type, optional
        The scope of the index. If provided, the index will
        be attached to a `xattree`-decorated class with the
        given name, if there is any above the current class
        in the hierarchy. The index value must be a string
        or a special `ROOT` class indicating the root node.
    """

    def wrap(cls):
        is_xattree = has(cls)
        if is_xattree and cls is cls.__xattree__[_CLASS]:
            raise TypeError("Class is already `xattree`-decorated.")

        orig_pre_init = getattr(cls, "__attrs_pre_init__", lambda _: None)
        orig_post_init = getattr(cls, "__attrs_post_init__", lambda _: None)

        def pre_init(self):
            orig_pre_init(self)
            setattr(self, _XATTREE_READY, False)

        def run_converters(self):
            converters = cls.__xattree__.get(_CONVERTERS, {})
            if not any(converters):
                return
            spec = cls.__xattree__[_SPEC]
            for name, converter in converters.items():
                if (value := self.__dict__.get(name, None)) is not None:
                    match converter:
                        case Converter():
                            if converter.takes_self and converter.takes_field:
                                self.__dict__[name] = converter.converter(
                                    value, self, spec.flat[name]
                                )
                            elif converter.takes_self:
                                self.__dict__[name] = converter.converter(value, self)
                            elif converter.takes_field:
                                self.__dict__[name] = converter.converter(value, spec.flat[name])
                            else:
                                self.__dict__[name] = converter.converter(value)
                        case f if callable(f):
                            self.__dict__[name] = converter(value)

        def run_validators(self):
            validators = cls.__xattree__.get(_VALIDATORS, {})
            if not any(validators):
                return
            spec = cls.__xattree__[_SPEC]
            for name, validator in validators.items():
                if (value := self.__dict__.get(name, None)) is not None:
                    for f in validator:
                        f(self, spec.flat[name], value)

        def post_init(self):
            run_converters(self)
            run_validators(self)
            orig_post_init(self)
            if getattr(self, _XATTREE_READY, False):
                # the instance might already be initialized if
                # the class inherits from a xattree base class
                return
            _init_tree(
                self,
                strict=self.strict,
                where=cls.__xattree__[_WHERE],
                index=cls.__xattree__[_INDEX],
            )
            setattr(self, _XATTREE_READY, True)

        converters = {}
        validators = {}

        def transformer(cls: type, fields: list[Attribute]) -> Iterator[Attribute]:
            def _transform_field(field: Attribute) -> Attribute:
                if field.name in _XTRA_ATTRS.keys():
                    raise ValueError(f"Field name '{field.name}' is reserved.")
                if (type_ := field.type) is None:
                    raise TypeError(f"Field '{field.name}' has no type.")

                args = get_args(type_)
                origin = get_origin(type_)
                iterable = isclass(origin) and issubclass(origin, Iterable)
                mapping = iterable and isclass(origin) and issubclass(origin, Mapping)
                is_child = (
                    has(type_)
                    or (mapping and attrs_has(args[-1]))
                    or (iterable and attrs_has(args[0]))
                )
                metadata = field.metadata.get(_PKG_NAME, {})
                converter = metadata.get(_CONVERTER, None)
                validator = metadata.get(_VALIDATOR, None)
                kind = metadata.get(_KIND, None)

                if kind == "array":
                    if converter is not None:
                        converters[field.name] = converter
                    if validator is not None:
                        validators[field.name] = validator
                    return field

                if is_child:
                    if converter is not None:
                        converters[field.name] = converter
                    if validator is not None:
                        validators[field.name] = validator

                    optional = False
                    default = field.default
                    if default is NOTHING:
                        optional = True
                        default = Factory(lambda: type_(**({} if iterable else {_STRICT: False})))
                    elif default is None and iterable:
                        raise ValueError("Child collection's default may not be None.")
                    metadata = field.metadata.copy() or {}
                    multi = ("dict" if mapping else "list" if iterable else "only",)
                    metadata[_PKG_NAME] = {
                        _KIND: "child",
                        _TYPE: type_,
                        _OPTIONAL: optional,
                        _MULTI: multi,
                    }
                    return Attribute(  # type: ignore
                        name=field.name,
                        default=default,
                        validator=None,
                        repr=field.repr,
                        cmp=None,
                        hash=field.hash,
                        eq=field.eq,
                        init=field.init,
                        inherited=field.inherited,  # type: ignore
                        metadata=metadata,
                        type=field.type,
                        converter=None,
                        kw_only=field.kw_only,
                        eq_key=field.eq_key,  # type: ignore
                        order=field.order,
                        order_key=field.order_key,  # type: ignore
                        on_setattr=field.on_setattr,
                        alias=field.alias,
                    )
                return Attribute(  # type: ignore
                    name=field.name,
                    default=field.default,
                    validator=field.validator if kind else (validator or field.validator),
                    repr=field.repr,
                    cmp=None,
                    hash=field.hash,
                    eq=field.eq,
                    init=field.init,
                    inherited=field.inherited,  # type: ignore
                    metadata=field.metadata,
                    type=field.type,
                    converter=field.converter if kind else (converter or field.converter),
                    kw_only=field.kw_only,
                    eq_key=field.eq_key,  # type: ignore
                    order=field.order,
                    order_key=field.order_key,  # type: ignore
                    on_setattr=field.on_setattr,
                    alias=field.alias,
                )

            # rename the datatree field
            xtra_attrs = _XTRA_ATTRS.copy()
            datatree = xtra_attrs.pop(_DATA)
            xtra_attrs[where] = datatree

            if is_xattree:
                fields = [f for f in fields if f.name not in xtra_attrs.keys()]

            attrs_ = [_transform_field(f) for f in fields]
            extra = [f(cls) if callable(f) else f for f in xtra_attrs.values()]
            return attrs_ + extra  # type: ignore

        old_setattr = cls.__setattr__

        def _setattr(self: Any, name: str, value: Any):
            where = cls.__xattree__[_WHERE]
            if not getattr(self, _XATTREE_READY, False) or name in [
                where,
                _XATTREE_READY,
            ]:
                self.__dict__[name] = value
                return
            tree = getattr(self, where)
            if set_xattr := _XTRA_SETTERS.get(name, None):
                return set_xattr(tree, name, value)
            spec = _get_xatspec(cls)
            if not (xat := spec.flat.get(name, None)):
                old_setattr(self, name, value)
            match xat:
                case Coord():
                    raise AttributeError(f"Cannot set dimension/coordinate '{name}'.")
                case Attr():
                    tree.attrs[xat.name] = value
                    setattr(self, where, tree)
                case Array():
                    tree[xat.name] = xr.DataArray(value, dims=xat.dims)
                    setattr(self, where, tree)
                case Child():
                    if getattr(value, "parent", None) is not None:
                        raise AttributeError(f"Child '{name}' already has a parent, can't set it.")

                    def drop_matching_children(node: xr.DataTree) -> xr.DataTree:
                        return node.filter(lambda c: not issubclass(type(c.attrs[_HOST]), xat.type))  # type: ignore

                    match xat.kind:
                        case "dict":
                            tree = drop_matching_children(tree)
                            new_nodes = {k: getattr(v, where) for k, v in value.items()}
                        case "list":
                            tree = drop_matching_children(tree)
                            new_nodes = {
                                f"{xat.name}{i}": getattr(v, where) for i, v in enumerate(value)
                            }
                        case _:
                            new_nodes = {xat.name: getattr(value, where)}

                    new_hosts = {k: v.attrs[_HOST] for k, v in new_nodes.items()}
                    old_nodes = dict(tree.children)
                    tree = tree.assign(old_nodes | new_nodes)
                    setattr(self, where, tree)
                    _bind_tree(self, children=self.children | new_hosts)

        cls.__attrs_pre_init__ = pre_init
        cls.__attrs_post_init__ = post_init
        cls = define(cls, slots=False, field_transformer=transformer)
        cls.__getattr__ = _getattr
        cls.__setattr__ = _setattr
        cls.__xattree__ = {
            _CLASS: cls,
            _WHERE: where,
            _INDEX: index,
            _INDEX_SCOPE: index_scope,
            _SPEC: _get_xatspec(cls),
            _CONVERTERS: converters,
            _VALIDATORS: validators,
        }
        return cls

    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)
