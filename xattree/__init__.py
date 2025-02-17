import builtins
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict, get_origin, overload

import numpy as np
from attr import Attribute, fields_dict
from attrs import NOTHING, Factory, cmp_using, define, field, has
from beartype.claw import beartype_this_package
from beartype.vale import Is
from numpy.typing import ArrayLike, NDArray
from xarray import DataArray, Dataset, DataTree

beartype_this_package()


DIM = "dim"
DIMS = "dims"
COORD = "coord"
SCOPE = "scope"
_WHERE = "where"
_WHERE_DEFAULT = "data"


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


class _XatSpec(TypedDict):
    dimensions: dict[str, _Dim]
    coordinates: dict[str, _Coord]
    scalars: dict[str, Attribute]
    arrays: dict[str, Attribute]
    children: dict[str, Any]


def _parse(spec: Mapping[str, Attribute]) -> _XatSpec:
    """Parse an `attrs` specification into a cat-tree specification."""
    dimensions = {}
    coordinates = {}
    scalars = {}
    arrays = {}
    children = {}

    for var in spec.values():
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
                assert dims is None
                assert has(t)
                children[var.name] = var
            case _:
                raise ValueError(f"Variable has no type: {var.name}")

    return _XatSpec(
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
    """Bind a cat tree to its parent and children."""
    name = getattr(self, where).name
    tree = getattr(self, where)
    children = children or {}

    # bind parent
    if parent:
        if name in parent.data:
            parent_tree = getattr(parent, where)
            parent_tree.update({name: tree})
            parent_tree.self = parent
        else:
            setattr(parent, where, parent_tree.assign({name: tree}))
            parent_tree.self = parent

        setattr(self, where, parent_tree[name])
        self.parent = parent

        # self node will have been displaced
        # in parent since node child updates
        # don't happen in-place.
        tree = getattr(self, where)

    # bind children
    for n, child in children.items():
        setattr(child, where, tree[n])
        tree[n].self = child
        tree[n].self.parent = self

    # give the data tree a reference to the instance
    # so it can be the class hierarchy's "backbone",
    # i.e. so that an instance can be accessed from
    # another instance's data tree in `getattribute`.
    # TODO: think thru the consequences here. how to
    # avoid memory leaks?
    tree.self = self
    setattr(self, where, tree)


def _init_tree(
    self: _HasAttrs,
    name: Optional[str] = None,
    parent: Optional[_HasAttrs] = None,
    children: Optional[Mapping[str, _HasAttrs]] = None,
    dimensions: Optional[Mapping[str, int]] = None,
    coordinates: Optional[Mapping[str, ArrayLike]] = None,
    strict: bool = True,
    where: str = _WHERE_DEFAULT,
):
    """
    Initialize a cat tree.

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
    spec = _parse(fields_dict(cls))
    scalars = {}
    arrays = {}
    coordinates = coordinates or {}
    dimensions = dimensions or {}
    children = children or {}

    def _yield_children():
        for var in spec["children"].values():
            if has(var.type):
                yield (var.name, self.__dict__.pop(var.name, None))

    children = dict(list(_yield_children())) | children

    def _yield_scalars():
        for var in spec["scalars"].values():
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
                    f"Class 'cls_name' array "
                    f"'{attr.name}' failed dim resolution: "
                    f"{', '.join(unresolved)}"
                )
            return None
        return _chexpand(value, shape)

    def _yield_arrays():
        if parent:
            parent_tree = getattr(parent, where)
            inherited_dims = dict(parent_tree.dims)
        else:
            inherited_dims = {}
        for var in spec["arrays"].values():
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
        for var in spec["arrays"].values():
            if not (coord := var.metadata.get("coord", None)):
                continue
            dim_name = coord.get("dim", var.name)
            if (
                array := _resolve_array(
                    var,
                    value=self.__dict__.pop(var.name, var.default),
                    strict=strict,
                )
            ) is not None:
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

    setattr(
        self,
        where,
        DataTree(
            Dataset(
                data_vars=arrays,
                coords=coordinates,
                attrs={
                    n: v
                    for n, v in scalars.items()  # if n not in dimensions
                },
            ),
            name=name or cls.__name__.lower(),
            children={n: getattr(c, where) for n, c in children.items()},
        ),
    )
    _bind_tree(self, parent=parent, children=children)


def _getattribute(self: _HasAttrs, name: str) -> Any:
    where = self.__xattree__["where"]
    match name:
        case _ if name == where:
            raise AttributeError
        case "parent":
            return None

    cls = type(self)
    spec = fields_dict(cls)
    tree = getattr(self, where)
    if spec.get(name, False):
        value = _get(tree, name, None)
        if isinstance(value, DataTree):
            return value.self
        if value is not None:
            return value

    raise AttributeError


def _pop_children(
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


def _yield_coords(
    scope: str, where: str = _WHERE_DEFAULT, **kwargs
) -> Iterator[tuple[str, tuple[str, Any]]]:
    for value in kwargs.values():
        cls = type(value)
        if not has(cls):
            continue
        spec = fields_dict(cls)
        tree = getattr(value, where)
        for n, var in spec.items():
            if coord := var.metadata.get("coord", None):
                if scope == coord.get("scope", None):
                    yield coord.get("dim", n), (n, tree.coords[n].data)
            if dim := var.metadata.get("dim", None):
                if scope == dim.get("scope", None):
                    coord_name = dim.get("coord", n)
                    yield coord_name, (n, tree.coords[coord_name].data)


@overload
def config(
    *,
    where: str = _WHERE_DEFAULT,
) -> Callable[[type[_HasAttrs]], type[_HasAttrs]]: ...


@overload
def config(maybe_cls: type[_HasAttrs]) -> type[_HasAttrs]: ...


def xattree(
    maybe_cls: Optional[type[_HasAttrs]] = None,
    *,
    where: str = _WHERE_DEFAULT,
) -> type[_HasAttrs]:
    """
    Make an `attrs`-based class a (node in a) cat tree.

    Notes
    -----
    For this to work, the class cannot use slots.
    """

    def validate(cls):
        spec = fields_dict(cls)
        reserved = ["name", "dims", "parent", "strict"]
        for name in reserved:
            if name in spec:
                raise ValueError(
                    f"A field may not be named '{name}', "
                    f"reserved names are: {', '.join(reserved)}"
                )

    def wrap(cls):
        cls = define(cls, slots=False)
        validate(cls)
        init_self = cls.__init__
        cls_name = cls.__name__.lower()

        def init(self, *args, **kwargs):
            name = kwargs.pop("name", cls_name)
            parent = args[0] if args and any(args) else None
            children, kwargs = _pop_children(self, **kwargs)
            dimensions = kwargs.pop("dims", {})
            coordinates = dict(list(_yield_coords(scope=cls_name, **children)))
            strict = kwargs.pop("strict", True)
            init_self(self, **kwargs)
            _init_tree(
                self,
                name=name,
                parent=parent,
                children=children,
                dimensions=dimensions,
                coordinates=coordinates,
                strict=strict,
                where=where,
            )
            cls.__xattree__ = {_WHERE: where}
            cls.__getattr__ = _getattribute
            # TODO override __setattr__ for mutations

        cls.__init__ = init
        return cls

    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)


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
        default=NOTHING,
        validator=validator,
        repr=repr,
        eq=eq or cmp_using(eq=np.array_equal),
        order=False,
        hash=False,
        init=False,
        metadata=metadata,
    )


def child(
    cls,
    default=NOTHING,
    validator=None,
    repr=True,
    eq=True,
    metadata=None,
):
    return field(
        type=cls,
        default=default or Factory(lambda: cls(strict=False)),
        validator=validator,
        repr=repr,
        eq=eq,
        order=False,
        hash=True,
        init=True,
        metadata=metadata,
    )
