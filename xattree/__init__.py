"""
Herd an unruly glaring of `attrs` classes into an orderly `xarray.DataTree`.
"""

import builtins
import types
from collections import ChainMap
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, MutableSequence
from datetime import datetime
from inspect import isclass
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
import xarray as xa
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
from xarray.core.types import Self

_PKG_NAME = "xattree"


class _XatTree(xa.DataTree):
    """Monkey-patch `DataTree` with a reference to a host object."""

    # DataTree is not yet a proper slotted class, it still has `__dict__`.
    # So monkey-patching is not strictly necessary yet, but it will be.
    # When it is, this will start enforcing no dynamic attributes. See
    #   - https://github.com/pydata/xarray/issues/9068
    #   - https://github.com/pydata/xarray/issues/9928
    __slots__ = ("_host",)

    def __init__(self, dataset=None, children=None, name=None, host=None):
        super().__init__(dataset=dataset, children=children, name=name)
        self._host = host  # TODO: weakref?

    def __copy__(self):
        new = super().__copy__()
        new._host = self._host
        return new

    def __deepcopy__(self, memo=None):
        new = super().__deepcopy__(memo)
        new._host = self._host
        return new

    def copy(self, *, inherit: bool = True, deep: bool = False) -> Self:
        new = super().copy(inherit=inherit, deep=deep)
        new._host = self._host
        return new


xa.DataTree = _XatTree  # type: ignore


class _XatList(MutableSequence):
    """Proxy a `DataTree`'s children of a given type through a list-like interface."""

    def __init__(self, tree: xa.DataTree, xat: "_Xattribute", where: str):
        self._tree = tree
        self._xat = xat
        self._where = where
        self._cache = self._build_cache()

    def _build_cache(self) -> list[Any]:
        return [
            c._host
            for c in self._tree.children.values()
            if issubclass(type(c._host), self._xat.type)  # type: ignore
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
        def _set(key, val):
            host = self._tree._host
            node = getattr(val, self._where)
            self._tree = self._tree.assign(dict(self._tree.children) | {key: node})
            setattr(host, self._where, self._tree)
            # TODO bind only once instead of for each child if multiple?
            _bind_tree(host, children=host.children | {key: node._host})

        if isinstance(index, slice):
            for i, v in enumerate(value):
                key = f"{self._xat.name}{index.start + i}"
                _set(key, v)
        else:
            key = f"{self._xat.name}{index}"
            _set(key, value)

        self._cache = self._build_cache()

    @overload
    def __delitem__(self, index: int) -> None: ...

    @overload
    def __delitem__(self, index: slice) -> None: ...

    def __delitem__(self, index: int | slice) -> None:
        if isinstance(index, slice):
            for i in range(index.start or 0, index.stop or len(self._cache)):
                key = f"{self._xat.name}{i}"
                del self._tree[key]
        else:
            key = f"{self._xat.name}{index}"
            del self._tree[key]
        self._cache = self._build_cache()

    def __iter__(self):
        return iter(self._cache)

    def __repr__(self):
        return list.__repr__(self._cache)

    def insert(self, index: int, value: Any):
        self.__setitem__(index, value)


class _XatDict(MutableMapping):
    """Proxy a `DataTree`'s children of a given type through a dict-like interface."""

    def __init__(self, tree: xa.DataTree, xat: "_Xattribute", where: str):
        self._tree = tree
        self._xat = xat
        self._where = where
        self._cache = self._build_cache()

    def _build_cache(self) -> dict[str, Any]:
        return {
            n: c._host
            for n, c in self._tree.children.items()
            if issubclass(type(c._host), self._xat.type)  # type: ignore
        }

    def __eq__(self, value):
        return self._cache == value

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, key: str) -> Any:
        return self._cache[key]

    def __setitem__(self, key: str, value: Any):
        host = self._tree._host
        node = getattr(value, self._where)
        self._tree = self._tree.assign(dict(self._tree.children) | {key: node})
        setattr(host, self._where, self._tree)
        _bind_tree(host, children=host.children | {key: node._host})
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


_Int = int | np.int32 | np.int64
_Numeric = int | float | np.int64 | np.float64
_Scalar = bool | _Numeric | str | Path | datetime
_KIND = "kind"
_NAME = "name"
_DIMS = "dims"
_SCOPE = "scope"
_SPEC = "spec"
_STRICT = "strict"
_TYPE = "type"
_OPTIONAL = "optional"
_CONVERTER = "converter"
_CONVERTERS = "converters"
_MULTI = "multi"
_VALIDATOR = "validator"
_VALIDATORS = "validators"
_WHERE = "where"
_WHERE_DEFAULT = "data"
_XATTREE_DUNDER = "__xattree__"
_XATTREE_READY = "_xattree_ready"
_XTRA_ATTRS = {
    "name": lambda cls: Attribute(  # type: ignore
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
    "dims": Attribute(  # type: ignore
        name="dims",
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
    "parent": Attribute(  # type: ignore
        name="parent",
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
    "children": Attribute(  # type: ignore
        name="children",
        default=Factory(dict),
        validator=None,
        repr=False,
        cmp=None,
        hash=False,
        eq=False,
        init=True,
        inherited=False,
        type=Mapping[str, Any],
    ),
    "strict": Attribute(  # type: ignore
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
_XTRA_ACCSSORS = {
    "name": lambda tree: tree.name,
    "dims": lambda tree: tree.dims,
    "parent": lambda tree: None if tree.is_root else tree.parent._host,
    "children": lambda tree: {n: c._host for n, c in tree.children.items()},
    "strict": lambda _: False,
}


def _chexpand(value: ArrayLike, shape: tuple[int]) -> Optional[NDArray]:
    value = np.array(value)
    if value.shape == ():
        return np.full(shape, value.item())
    if value.shape != shape:
        raise ValueError(f"Shape mismatch, got {value.shape}, expected {shape}")
    return value


@define
class _Xattribute:
    name: str
    default: Optional[Any] = None
    optional: bool = False
    type: Optional["type"] = None
    converter: Optional[Callable] = None
    metadata: Optional[dict[str, Any]] = None


@define
class _Attr(_Xattribute):
    pass


@define
class _Array(_Xattribute):
    dims: Optional[tuple[str, ...]] = None
    dtype: Optional["type"] = None


@define
class _Coord(_Xattribute):
    alias: Optional[str] = None
    path: Optional[str] = None
    scope: Optional[str] = None
    from_dim: Optional[bool] = False


_ChildKind = Literal["only", "list", "dict"]


@define
class _Child(_Xattribute):
    type: Optional["type"] = None
    kind: _ChildKind = "only"


@define
class _XatSpec:
    attrs: dict[str, _Attr]
    arrays: dict[str, _Array]
    coords: dict[str, _Coord]
    children: dict[str, _Child]

    @property
    def flat(self) -> MutableMapping[str, _Xattribute]:
        return ChainMap(self.attrs, self.arrays, self.coords, self.children)  # type: ignore


def _get_xatspec(cls: type) -> _XatSpec:
    """Extract a `xattree` specification from a given class."""
    cls_name = cls.__name__

    def __xatspec(fields: dict) -> _XatSpec:
        attributes = {}
        arrays = {}
        coords = {}
        children = {}

        def _register_nested_dims(child_spec: _Child, path=None):
            if child_spec.type is None:
                return
            for child in (spec := _get_xatspec(child_spec.type)).children.values():
                if child.type:
                    _register_nested_dims(
                        child, path=f"{path}/{child.name}" if path else child.name
                    )
            cls_name_l = cls_name.lower()
            for alias, coord in spec.coords.items():
                if coord.scope is ROOT or coord.scope == cls_name_l:
                    coords[alias] = evolve(
                        coord, path=f"{path}/{child_spec.name}" if path else child_spec.name
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
                    name = xatmeta.get(_NAME, None) or field.name
                    if origin in (Union, types.UnionType):
                        if args[-1] is types.NoneType:  # Optional
                            is_optional = True
                            type_ = args[0]
                        else:
                            raise TypeError(f"Dim must have a concrete type: {field.name}")
                    if not (isclass(type_) and issubclass(type_, _Int)):
                        raise TypeError(f"Dim '{field.name}' must be an integer")
                    coords[name] = _Coord(
                        alias=name if name != field.name else None,
                        name=field.name,
                        default=field.default,
                        optional=is_optional,
                        scope=xatmeta.get(_SCOPE, None),
                        from_dim=True,
                        metadata=metadata,
                    )
                    attributes[field.name] = _Attr(
                        name=field.name,
                        default=field.default,
                        optional=is_optional,
                        metadata=metadata,
                    )
                case "coord":
                    if not (isclass(origin) and issubclass(origin, np.ndarray)):
                        raise TypeError(f"Coord '{field.name}' must be an array type")
                    coords[field.name] = _Coord(
                        name=field.name,
                        default=field.default,
                        optional=is_optional,
                        scope=xatmeta.get(_SCOPE, None),
                        metadata=metadata,
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
                    arrays[field.name] = _Array(
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
                    child_kind: _ChildKind | None = None
                    is_child = False
                    is_optional = False
                    iterable = isclass(origin) and issubclass(origin, Iterable)
                    mapping = iterable and issubclass(origin, Mapping)
                    if origin in (Union, types.UnionType):
                        if args[-1] is types.NoneType:  # Optional
                            is_optional = True
                            origin = None
                            type_ = args[0]
                    elif not origin and attrs_has(type_):
                        is_child = True
                        child_kind = "only"
                    elif iterable or mapping:
                        match len(args):
                            case 1:
                                type_ = args[0]
                                if attrs_has(type_):
                                    is_child = True
                                    child_kind = "list"
                            case 2:
                                type_ = args[1]
                                if args[0] is str and attrs_has(type_):
                                    is_child = True
                                    child_kind = "dict"
                    if is_child:
                        child = _Child(
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
                        attributes[field.name] = _Attr(
                            name=field.name,
                            default=field.default,
                            optional=is_optional,
                            metadata=metadata,
                        )

        return _XatSpec(attrs=attributes, arrays=arrays, coords=coords, children=children)

    if (meta := getattr(cls, _XATTREE_DUNDER, None)) and (spec := meta.get(_SPEC, None)):
        return spec
    return __xatspec(fields_dict(cls))


def _bind_tree(
    self: Any,
    parent: Any = None,
    children: Optional[Mapping[str, Any]] = None,
    where: str = _WHERE_DEFAULT,
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
        parent_tree = getattr(parent, where)
        multi = cls.__xattree__.get(_MULTI, None)
        anon = name == cls.__name__.lower()
        match multi:
            case "list":
                items = {n: c for n, c in parent_tree.children.items()}
                same_type = {n: c for n, c in items.items() if type(c._host) is cls}
                name = f"{name}{len(same_type)}"
                if anon:
                    new = items | {name: tree}
                else:
                    new = {name: tree}
                if name in parent.data:
                    parent_tree.update(new)
                else:
                    parent_tree = parent_tree.assign(new)
            case "dict" | True:
                items = {n: c for n, c in parent_tree.children.items()}
                if anon:
                    new = items | {name: tree}
                else:
                    new = {name: tree}
                if name in parent.data:
                    parent_tree.update(new)
                else:
                    parent_tree = parent_tree.assign(new)
            case False | None:
                if name in parent.data:
                    parent_tree.update({name: tree})
                else:
                    parent_tree = parent_tree.assign({name: tree, **parent_tree.children})
        parent_tree._host = parent
        setattr(parent, where, parent_tree)
        tree = parent_tree[name]
        setattr(self, where, tree)

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


def _init_tree(self: Any, strict: bool = True, where: str = _WHERE_DEFAULT):
    """
    Initialize a `xattree`-decorated class instance's `DataTree`.

    Notes
    -----
    This method must run after the default `__init__()`.

    The tree is built from the class' `attrs` fields, i.e.
    spirited from the instance's `__dict__` into the tree,
    which is added as an attribute whose name is `where`.
    `__dict__` is emptyish after this method runs except
    the data tree. Field access is proxied to the tree.

    The decorated class cannot use slots for this to work.
    """
    cls = type(self)
    cls_name = cls.__name__
    name = self.__dict__.pop("name", cls_name.lower())
    parent = self.__dict__.pop("parent", None)
    explicit_dims = self.__dict__.pop("dims", None) or {}
    xatspec = _get_xatspec(cls)

    def _yield_children() -> Iterator[tuple[str, Any]]:
        for child in self.__dict__.pop("children", {}).values():
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
        for xat_name, xat in xatspec.attrs.items():
            yield (xat_name, self.__dict__.pop(xat_name, xat.default))

    children = dict(list(_yield_children()))
    attributes = dict(list(_yield_attrs()))

    def _resolve_array(
        xat: _Xattribute, value: ArrayLike, strict: bool = False, **dims
    ) -> Optional[NDArray]:
        dims = dims or {}
        match xat:
            case _Coord():
                if xat.default is None or not isinstance(xat.default, _Scalar):
                    raise CannotExpand(
                        f"Class '{cls_name}' coord array '{xat.name}'"
                        f"paired with dim '{xat.name}' can't expand "
                        f"without a scalar default dimension size."
                    )
                return _chexpand(value, (xat.default,))
            case _Array():
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
                    value = value or xat.default  # type: ignore
                    if value is None:
                        return None  # type: ignore
                    return None if any(unresolved) else _chexpand(value, shape)
                value = np.array(value)
                if xat.dims and value.ndim != len(shape):
                    raise ValueError(
                        f"Class '{cls_name}' array '{xat.name}' "
                        f"expected {len(shape)} dims, got {value.ndim}"
                    )
                return value
        return None

    def _find_dim_or_coord(
        children: Mapping[str, Any], coord: _Coord
    ) -> Optional[Union[ArrayLike, _Scalar]]:
        if not coord.path:
            return None
        coord_name = coord.alias or coord.name
        child_name, _, path = coord.path.partition("/")
        match len(path):
            case 1:
                if (child := children.get(child_name, None)) is None:
                    return None
                child_node = getattr(child, where)
                if coord.from_dim:
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
                        if coord.from_dim
                        else target_node.coords[coord_name].data
                    )
                except KeyError:
                    raise KeyError(
                        f"Coord '{coord_name}' declared but not found in "
                        f"scope '{child_name}', is it initialized? If a "
                        f"derived dim/coord, make sure you're using the "
                        f"__attrs_post_init__() method to initialize it."
                    )

    dimensions = {}

    def _yield_coords() -> Iterator[tuple[str, tuple[str, NDArray]]]:
        # register inherited dimension sizes so we can expand arrays
        if parent:
            parent_tree: xa.DataTree = getattr(parent, where)
            for coord in parent_tree.coords.values():
                dimensions[coord.dims[0]] = coord.data.size

        # yield coord arrays, expanding from dim sizes if necessary
        known_dims = dimensions | explicit_dims
        for alias, coord in xatspec.coords.items():
            value = self.__dict__.pop(coord.name, None)
            if value is None or value is NOTHING:
                value = known_dims.get(alias, None) or _find_dim_or_coord(children, coord)
            if value is None or value is NOTHING:
                value = coord.default
            if value is None or value is NOTHING:
                value = attributes.get(coord.name, None)
            if value is None or value is NOTHING:
                value = attributes.get(alias, None)
            if value is None:
                continue
            if isinstance(value, _Scalar):
                match type(value):
                    case builtins.int | builtins.float | np.number:
                        step = 1
                        start = 0
                    case _:
                        raise ValueError("Dim size must be numeric.")
                array: np.ndarray = np.arange(start, value, step)
            else:
                array = np.array(value)
            dimensions[alias] = len(array)
            yield (alias, (alias, array))

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

    setattr(
        self,
        where,
        _XatTree(
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


def _getattr(self: Any, name: str) -> Any:
    cls = type(self)
    if name == (where := cls.__xattree__[_WHERE]):
        raise AttributeError
    if name == _XATTREE_READY:
        return False
    tree = cast(xa.DataTree, getattr(self, where, None))
    if access_xattr := _XTRA_ACCSSORS.get(name, None):
        return access_xattr(tree)
    spec = _get_xatspec(cls)
    if xat := spec.flat.get(name, None):
        match xat:
            case _Coord():
                if xat.from_dim:
                    try:
                        return tree.dims[xat.name]
                    except KeyError:
                        return tree.attrs[name]
                return tree.coords[xat.name].data
            case _Attr():
                return tree.attrs[xat.name]
            case _Array():
                try:
                    return tree[xat.name]
                except KeyError:
                    return None
            case _Child():
                match xat.kind:
                    case "dict":
                        return _XatDict(tree, xat, where)
                    case "list":
                        return _XatList(tree, xat, where)
                    case "only":
                        if (child := tree.children.get(xat.name, None)) is not None:
                            return child._host
                        return None
            case _:
                raise TypeError(
                    f"Field '{name}' is not a dimension, coordinate, "
                    "attribute, array, or child variable"
                )

    raise AttributeError


def _setattr(self: Any, name: str, value: Any):
    cls = type(self)
    cls_name = cls.__name__
    where = cls.__xattree__[_WHERE]
    if not getattr(self, _XATTREE_READY, False) or name in [
        where,
        _XATTREE_READY,
    ]:
        self.__dict__[name] = value
        return
    spec = _get_xatspec(cls)
    if not (xat := spec.flat.get(name, None)):
        raise AttributeError(f"{cls_name} has no field {name}")
    tree = getattr(self, where)
    match xat:
        case _Coord():
            raise AttributeError(f"Cannot set dimension/coordinate '{name}'.")
        case _Attr():
            tree.attrs[xat.name] = value
            setattr(self, where, tree)
        case _Array():
            tree[xat.name] = value
            setattr(self, where, tree)
        case _Child():

            def drop_matching_children(node: xa.DataTree) -> xa.DataTree:
                return node.filter(lambda c: not issubclass(type(c._host), xat.type))  # type: ignore

            # DataTree.assign() replaces only the entries you provide it,
            # but we need to replace the entire subtree to make sure each
            # node's host reference survives. TODO: why?? overriding copy
            # and deepcopy should be enough?
            match xat.kind:
                case "dict":
                    tree = drop_matching_children(tree)
                    new_nodes = {k: getattr(v, where) for k, v in value.items()}
                case "list":
                    tree = drop_matching_children(tree)
                    new_nodes = {f"{xat.name}{i}": getattr(v, where) for i, v in enumerate(value)}
                case _:
                    new_nodes = {xat.name: getattr(value, where)}
            new_hosts = {k: v._host for k, v in new_nodes.items()}
            old_nodes = dict(tree.children)
            tree = tree.assign(old_nodes | new_nodes)
            setattr(self, where, tree)
            _bind_tree(self, children=self.children | new_hosts)


def field(
    default=NOTHING,
    validator=None,
    repr=True,
    eq=True,
    init=True,
    metadata=None,
    converter=None,
):
    """Create a field."""
    metadata = metadata or {}
    metadata[_PKG_NAME] = {_KIND: None}  # unknown, infer later
    return attrs_field(
        default=default,
        validator=validator,
        repr=repr,
        eq=eq,
        order=False,
        hash=True,
        init=init,
        metadata=metadata,
        converter=converter,
    )


def dim(
    name=None,
    scope=None,
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
        _NAME: name,
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
    repr=True,
    eq=None,
    metadata=None,
    converter=None,
):
    """Create an array field."""
    dims = dims if isinstance(dims, Iterable) else tuple()
    if not any(dims) and isinstance(default, _Scalar):
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
    """Check whether `field` is a `xattree` attribute/field."""
    return _PKG_NAME in field.metadata


def has_xats(cls) -> bool:
    """Check whether `cls` is a `xattree`."""
    return hasattr(cls, _XATTREE_DUNDER)


def fields_dict(cls, just_yours: bool = True) -> dict[str, Attribute]:
    """
    Get the field dict for a class. By default, only your
    attributes are included, none of the special attributes
    attached by `xattree`. To include those, set `just_yours=False`.
    """
    return {
        n: f
        for n, f in attrs_fields_dict(cls).items()
        if not just_yours or n not in _XTRA_ATTRS.keys()
    }


def fields(cls, just_yours: bool = True) -> list[Attribute]:
    """
    Get the field list for a class. By default, only your
    attributes are included, none of the special attributes
    attached by `xattree`. To include those, set `just_yours=False`.
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


@dataclass_transform(field_specifiers=(attrs_field, field, dim, coord, array))
def xattree(
    maybe_cls: Optional[type[Any]] = None,
    *,
    where: str = _WHERE_DEFAULT,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Make an `attrs`-based class a (node in a) `xattree`."""

    def wrap(cls):
        if has_xats(cls):
            raise TypeError("Class is already a `xattree`.")

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
            for n, c in converters.items():
                if (val := self.__dict__.get(n, None)) is not None:
                    match c:
                        case Converter():
                            if c.takes_self and c.takes_field:
                                self.__dict__[n] = c.converter(val, self, spec.flat[n])
                            elif c.takes_self:
                                self.__dict__[n] = c.converter(val, self)
                            elif c.takes_field:
                                self.__dict__[n] = c.converter(val, spec.flat[n])
                            else:
                                self.__dict__[n] = c.converter(val)
                        case f if callable(f):
                            self.__dict__[n] = c(val)

        def run_validators(self):
            validators = cls.__xattree__.get(_VALIDATORS, {})
            if not any(validators):
                return
            spec = cls.__xattree__[_SPEC]
            for n, v in validators.items():
                if (val := self.__dict__.get(n, None)) is not None:
                    for validate in v:
                        validate(self, spec.flat[n], val)

        def post_init(self):
            run_converters(self)
            run_validators(self)
            orig_post_init(self)
            _init_tree(self, strict=self.strict, where=cls.__xattree__[_WHERE])
            setattr(self, _XATTREE_READY, True)

        converters = {}
        validators = {}

        def transformer(cls: type, fields: list[Attribute]) -> Iterator[Attribute]:
            def _transform_field(field: Attribute) -> Attribute:
                if field.name in _XTRA_ATTRS.keys():
                    raise ValueError(f"Field name '{field.name}' is reserved.")

                if (type_ := field.type) is None:
                    return field
                args = get_args(type_)
                origin = get_origin(type_)
                iterable = isclass(origin) and issubclass(origin, Iterable)
                mapping = iterable and isclass(origin) and issubclass(origin, Mapping)
                metadata = field.metadata.get(_PKG_NAME, {})
                if (converter := metadata.get(_CONVERTER, None)) is not None:
                    converters[field.name] = converter
                if (validator := metadata.get(_VALIDATOR, None)) is not None:
                    validators[field.name] = validator
                else:
                    if iterable or mapping:
                        # TODO: do we want to deep validate collections? if we do,
                        # doing it exhaustively is not feasible. we want something like
                        # beartype which randomly chooses an element to validate instead.
                        pass
                    else:
                        # TODO type validation?
                        # validators[field.name] = [attrs.validators.instance_of(origin or type_)]
                        pass

                if not (
                    attrs_has(type_)
                    or (mapping and attrs_has(args[-1]))
                    or (iterable and attrs_has(args[0]))
                ):
                    return field

                # detect and register child fields. can be singles or collections.
                optional = False
                default = field.default
                if default is NOTHING:
                    optional = True
                    default = Factory(lambda: type_(**({} if iterable else {_STRICT: False})))
                elif default is None and iterable:
                    raise ValueError("Child collection's default may not be None.")
                metadata = field.metadata.copy() or {}
                metadata[_PKG_NAME] = {
                    _KIND: "child",
                    _NAME: field.name,
                    _TYPE: type_,
                    _OPTIONAL: optional,
                    _MULTI: "dict" if mapping else "list" if iterable else "only",
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

            attrs_ = [_transform_field(f) for f in fields]
            extra = [f(cls) if callable(f) else f for f in _XTRA_ATTRS.values()]
            return attrs_ + extra  # type: ignore

        cls.__attrs_pre_init__ = pre_init
        cls.__attrs_post_init__ = post_init
        cls = define(cls, slots=False, field_transformer=transformer)
        cls.__getattr__ = _getattr
        cls.__setattr__ = _setattr
        cls.__xattree__ = {
            _WHERE: where,
            _SPEC: _get_xatspec(cls),
            _CONVERTERS: converters,
            _VALIDATORS: validators,
        }
        return cls

    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)
