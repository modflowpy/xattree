import builtins
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Optional,
    TypedDict,
    TypeVar,
    dataclass_transform,
    get_origin,
    overload,
)

import numpy as np
from attrs import (
    NOTHING,
    Attribute,
    Factory,
    cmp_using,
    define,
    field,
    fields_dict,
    has,
)
from beartype.claw import beartype_this_package
from beartype.vale import Is
from numpy.typing import ArrayLike, NDArray
from xarray import DataArray, Dataset, DataTree

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


_Numeric = int | float | np.int64 | np.float64
"""A numeric value."""

_Scalar = bool | _Numeric | str | Path
"""A scalar value."""

_Array = list | np.ndarray
"""An array value."""

_HasAttrs = Annotated[object, Is[lambda obj: has(type(obj))]]
"""`attrs` based class instances."""


DIM = "dim"
DIMS = "dims"
COORD = "coord"
SCOPE = "scope"
_WHERE = "where"
_WHERE_DEFAULT = "data"
_READY = "ready"
_XATTREE_READY = "_xattree_ready"
_XATTREE_FIELDS = {
    "name": lambda cls: Attribute(
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
    "parent": Attribute(
        name="parent",
        default=None,
        validator=None,
        repr=True,
        cmp=None,
        hash=False,
        eq=False,
        init=True,
        inherited=False,
        type=_HasAttrs,
    ),
    "strict": Attribute(
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


def _get(
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


class _Dim(TypedDict):
    name: str
    scope: Optional[str]
    attr: Optional[Attribute]


class _Coord(TypedDict):
    name: str
    scope: Optional[str]
    attr: Optional[Attribute]


class _TreeSpec(TypedDict):
    dimensions: dict[str, _Dim]
    coordinates: dict[str, _Coord]
    scalars: dict[str, Attribute]
    arrays: dict[str, Attribute]
    children: dict[str, Any]


def _parse(spec: Mapping[str, Attribute]) -> _TreeSpec:
    """Parse an `attrs` specification into a tree specification."""
    dimensions = {}
    coordinates = {}
    scalars = {}
    arrays = {}
    children = {}

    for var in spec.values():
        if var.name in _XATTREE_FIELDS.keys():
            continue

        dim = var.metadata.get("dim", None)
        dims = var.metadata.get("dims", None)
        coord = var.metadata.get("coord", None)

        if dim and coord:
            raise ValueError(
                f"Variable '{var.name}' cannot have "
                f"both 'dim' and 'coord' metadata."
            )

        match var.type:
            # array
            case t if t and issubclass(get_origin(t) or object, _Array):
                arrays[var.name] = var
                if coord:
                    dim_name = (
                        coord.get("dim", var.name)
                        if isinstance(coord, dict)
                        else var.name
                    )
                    scope = coord.get("scope", None)
                    dimensions[dim_name] = _Dim(
                        name=dim_name,
                        scope=scope,
                    )
                    coordinates[dim_name] = _Coord(
                        name=var.name, scope=scope, attr=var
                    )
            # scalar
            case t if t and issubclass(t, _Scalar):
                assert dims is None
                scalars[var.name] = var
                if dim:
                    is_dict = isinstance(dim, dict)
                    dimensions[var.name] = _Dim(
                        name=var.name,
                        scope=dim.get("scope", None) if is_dict else None,
                        attr=var,
                    )
                    if not is_dict:
                        continue
                    coordinates[var.name] = _Coord(
                        name=dim.get("coord", var.name),
                        scope=dim.get("scope", None),
                    )
            # child
            case t if t:
                assert dims is None and has(t)
                children[var.name] = var
            case _:
                raise ValueError(f"Variable has no type: {var.name}")

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
            parent_tree.self = parent
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
    spec = fields_dict(cls)
    catspec = _parse(spec)
    scalars = {}
    arrays = {}
    parent = self.__dict__.pop("parent", None)

    def _yield_children():
        for var in catspec["children"].values():
            if has(var.type):
                if child := self.__dict__.pop(var.name, None):
                    yield (var.name, child)

    children = dict(list(_yield_children()))

    def _yield_coords(scope, **objs) -> Iterator[tuple[str, tuple[str, Any]]]:
        for obj in objs.values():
            if not has(cls := type(obj)):
                continue
            spec = fields_dict(cls)
            tree = getattr(obj, where)
            for n, var in spec.items():
                if coord := var.metadata.get("coord", None):
                    if scope == coord.get("scope", None):
                        yield coord.get("dim", n), (n, tree.coords[n].data)
                if dim := var.metadata.get("dim", None):
                    if scope == dim.get("scope", None):
                        coord_name = dim.get("coord", n)
                        yield coord_name, (n, tree.coords[coord_name].data)

    coordinates = dict(list(_yield_coords(scope=cls_name, **children)))
    dimensions = {}

    def _yield_scalars():
        for var in catspec["scalars"].values():
            yield (var.name, self.__dict__.pop(var.name, var.default))

    scalars = dict(list(_yield_scalars()))

    def _resolve_array(
        attr: Attribute,
        value: ArrayLike,
        strict: bool = False,
        **kwargs,
    ) -> tuple[Optional[NDArray], Optional[dict[str, NDArray]]]:
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
        shape = [kwargs.pop(dim, dim) for dim in dims]
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

    def _yield_arrays():
        inherited_dims = dict(getattr(parent, where).dims) if parent else {}
        for var in catspec["arrays"].values():
            dims = var.metadata.get("dims", None)
            if var.metadata.get("coord", False):
                continue
            if (
                array := _resolve_array(
                    var,
                    value=self.__dict__.pop(var.name, var.default),
                    strict=strict,
                    **(inherited_dims | dimensions),
                )
            ) is not None:
                yield (var.name, (dims, array) if dims else array)

    arrays = dict(list(_yield_arrays()))

    def _yield_coords():
        # coords are either explicitly provided
        # arrays, inferred/expanded from local
        # dims, or inherited from parent tree
        if parent:
            parent_tree = getattr(parent, where)
            for coord_name, coord in parent_tree.coords.items():
                yield (coord_name, (coord.dims, coord.data))
        for var in catspec["arrays"].values():
            if not (coord := var.metadata.get("coord", None)):
                continue
            if (
                array := _resolve_array(
                    var,
                    value=self.__dict__.pop(var.name, var.default),
                    strict=strict,
                )
            ) is not None:
                yield (var.name, (coord.get("dim", var.name), array))
        for scalar_name, scalar in scalars.items():
            if scalar_name not in catspec["dimensions"]:
                continue
            coord = catspec["coordinates"][scalar_name]
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
                (scalar_name, np.arange(start, scalar, step)),
            )

    coordinates = dict(list(_yield_coords())) | coordinates

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

    # ready = getattr(self, cls.__xattree__[_READY], False)
    # if not ready:
    #     raise AttributeError
    tree = getattr(self, where, None)
    match name:
        case "name":
            return tree.name
        case "parent":
            return None if tree.is_root else tree.parent.self
        case "children":
            return {n: c.self for n, c in tree.children.items()}

    spec = fields_dict(cls)
    if spec.get(name, False):
        value = _get(tree, name, None)
        if isinstance(value, DataTree):
            return value.self
        if value is not None:
            return value

    raise AttributeError


def _setattribute(self: _HasAttrs, name: str, value: Any):
    cls = type(self)
    ready = cls.__xattree__[_READY]
    where = cls.__xattree__[_WHERE]
    if not getattr(self, ready, False) or name == ready or name == where:
        self.__dict__[name] = value
        return
    spec = fields_dict(cls)
    if not (attr := spec.get(name, None)):
        raise AttributeError(f"{cls.__name__} has no attribute {name}")
    match attr.type:
        case t if has(t):
            children = self.children | {attr.name: getattr(value, where).self}
            _bind_tree(self, children=children)
            # setattr(
            #     self,
            #     where,
            #     getattr(self, where).assign(
            #         {attr.name: getattr(value, where)}
            #     ),
            # )
        case t if (origin := get_origin(t)) and issubclass(origin, _Array):
            self.data.update({attr.name: value})
        case t if not origin and issubclass(attr.type, _Scalar):
            self.data.attrs[attr.name] = value


def dim(
    coord=None,
    scope=None,
    default=NOTHING,
    validator=None,
    repr=True,
    eq=True,
    metadata=None,
):
    metadata = metadata or {}
    metadata[DIM] = {COORD: coord, SCOPE: scope}
    return field(
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
    default=NOTHING,
    validator=None,
    repr=True,
    eq=True,
    metadata=None,
):
    metadata = metadata or {}
    metadata[COORD] = {DIM: dim, SCOPE: scope}
    return field(
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
    dims=None,
    default=NOTHING,
    validator=None,
    repr=True,
    eq=None,
    metadata=None,
):
    metadata = metadata or {}
    metadata[DIMS] = dims

    def any_dims():
        if dims is None:
            return False
        if isinstance(dims, Iterable):
            return any(dims)
        return False

    if isinstance(default, _Scalar) and not any_dims():
        raise CannotExpand(
            "Can't have scalar default if dims are not provided."
        )
    return field(
        default=default,
        validator=validator,
        repr=repr,
        eq=eq or cmp_using(eq=np.array_equal),
        order=False,
        hash=False,
        init=True,
        metadata=metadata,
    )


def child(
    cls,
    default=None,
    validator=None,
    repr=True,
    eq=True,
    metadata=None,
):
    return field(
        default=default or Factory(lambda: cls(strict=False)),
        validator=validator,
        repr=repr,
        eq=eq,
        order=False,
        hash=True,
        init=True,
        metadata=metadata,
    )


T = TypeVar("T")


@overload
def xattree(
    *,
    where: str = _WHERE_DEFAULT,
) -> Callable[[type[T]], type[T]]: ...


@overload
def xattree(maybe_cls: type[T]) -> type[T]: ...


@dataclass_transform(field_specifiers=(field, dim, coord, array, child))
def xattree(
    maybe_cls: Optional[type[_HasAttrs]] = None,
    *,
    where: str = _WHERE_DEFAULT,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Make an `attrs`-based class a (node in a) `xattree`."""

    def wrap(cls):
        def pre_init(self):
            setattr(self, cls.__xattree__[_READY], False)

        def post_init(self):
            _init_tree(
                self, strict=self.strict, where=cls.__xattree__["where"]
            )
            setattr(self, cls.__xattree__[_READY], True)

        def transformer(cls: type, fields: list[Attribute]) -> list[Attribute]:
            return fields + [
                f(cls) if isinstance(f, Callable) else f
                for f in _XATTREE_FIELDS.values()
            ]

        cls.__attrs_pre_init__ = pre_init
        cls.__attrs_post_init__ = post_init
        cls = define(
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
