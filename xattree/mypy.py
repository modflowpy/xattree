from functools import partial
from typing import Final

from attrs import NOTHING
from mypy.nodes import MDEF, Context, SymbolTableNode, Var
from mypy.plugin import ClassDefContext, Plugin
from mypy.plugins.attrs import (
    ATTRS_INIT_NAME,
    Attribute,
    MethodAdder,
    _add_attrs_magic_attribute,
    _add_empty_metadata,
    _add_init,
    _add_match_args,
    _add_order,
    _add_slots,
    _analyze_class,
    _determine_eq_order,
    _get_decorator_bool_argument,
    _get_decorator_optional_bool_argument,
    _get_frozen,
    _make_frozen,
    _remove_hashability,
)
from mypy.plugins.attrs import (
    attr_class_maker_callback_impl as attrs_class_maker,
)
from mypy.plugins.attrs import (
    attr_class_makers as attrs_class_makers,
)
from mypy.types import AnyType, TypeOfAny

from xattree import _XTRA_ATTRS

attr_dataclass_makers: Final = {"xattree.xattree"}
attr_class_makers: Final = {"xattree.xattree"}
attr_attrib_makers: Final = {"xattree.field", "xattree.dim", "xattree.coord", "xattree.array"}


def python_type_to_mypy_type(py_type, ctx):
    # Handle builtins
    if py_type is bool:
        return ctx.api.named_type_or_none("builtins.bool")
    elif py_type is int:
        return ctx.api.named_type_or_none("builtins.int")
    elif py_type is float:
        return ctx.api.named_type_or_none("builtins.float")
    elif py_type is str:
        return ctx.api.named_type_or_none("builtins.str")
    elif py_type is dict:
        return ctx.api.named_type_or_none("builtins.dict")
    elif py_type is list:
        return ctx.api.named_type_or_none("builtins.list")
    elif hasattr(py_type, "__module__") and hasattr(py_type, "__name__"):
        # Try to resolve custom types
        fq_name = f"{py_type.__module__}.{py_type.__name__}"
        return ctx.api.named_type_or_none(fq_name)
    return AnyType(TypeOfAny.explicit)


def find_attribute_line_column(ctx: ClassDefContext, attr_name: str):
    for stmt in ctx.cls.defs.body:
        # Look for assignments like "foo: int = 1" or "foo = 1"
        if hasattr(stmt, "lvalues"):
            for lvalue in stmt.lvalues:
                if hasattr(lvalue, "name") and lvalue.name == attr_name:
                    return stmt.line, stmt.column
        if hasattr(stmt, "name") and stmt.name == attr_name:
            return stmt.line, stmt.column
    # fallback: use class line/column
    return ctx.cls.line, ctx.cls.column


def attr_to_mypy(attr, ctx):
    mypy_type = python_type_to_mypy_type(attr.type, ctx)
    return Attribute(
        name=attr.name,
        alias=attr.alias,
        info=mypy_type.type,
        has_default=attr.default is not NOTHING,
        init=attr.init,
        kw_only=attr.kw_only,
        converter=attr.converter,
        context=Context(*find_attribute_line_column(ctx, attr.name)),
        init_type=mypy_type,
    )


def xattree_class_maker(ctx: ClassDefContext) -> bool:
    # lifted from mypy.plugins.attrs with some modifications

    info = ctx.cls.info

    init = _get_decorator_bool_argument(ctx, "init", True)
    frozen = _get_frozen(ctx, False)
    order = _determine_eq_order(ctx)
    slots = _get_decorator_bool_argument(ctx, "slots", False)

    auto_attribs = _get_decorator_optional_bool_argument(ctx, "auto_attribs", True)
    kw_only = _get_decorator_bool_argument(ctx, "kw_only", False)
    match_args = _get_decorator_bool_argument(ctx, "match_args", True)

    for super_info in ctx.cls.info.mro[1:-1]:
        if "attrs_tag" in super_info.metadata and "attrs" not in super_info.metadata:
            # Super class is not ready yet. Request another pass.
            return False

    attributes = _analyze_class(ctx, auto_attribs, kw_only)

    # collect and convert extra attributes to mypy representations
    xtra_attrs = {n: f(ctx.cls) if callable(f) else f for n, f in _XTRA_ATTRS.items()}
    xtra_attrs = [attr_to_mypy(attr, ctx) for attr in xtra_attrs.values()]

    # attach extra attributes to the symbol table
    for attr in xtra_attrs:
        if attr.name not in info.names:
            var = Var(attr.name, attr.init_type)
            var.info = info
            info.names[attr.name] = SymbolTableNode(MDEF, var)

    # Check if attribute types are ready.
    for attr in attributes:
        node = info.get(attr.name)
        if node is None:
            # This name is likely blocked by some semantic analysis error that
            # should have been reported already.
            _add_empty_metadata(info)
            return True

    _add_attrs_magic_attribute(ctx, [(attr.name, info[attr.name].type) for attr in attributes])
    if slots:
        _add_slots(ctx, attributes)
    if match_args and ctx.api.options.python_version[:2] >= (3, 10):
        # `.__match_args__` is only added for python3.10+, but the argument
        # exists for earlier versions as well.
        _add_match_args(ctx, attributes)

    # Save the attributes so that subclasses can reuse them.
    ctx.cls.info.metadata["attrs"] = {
        "attributes": [attr.serialize() for attr in attributes],
        "frozen": frozen,
    }

    attributes.extend(xtra_attrs)

    adder = MethodAdder(ctx)
    # If  __init__ is not being generated, attrs still generates it as __attrs_init__ instead.
    _add_init(ctx, attributes, adder, "__init__" if init else ATTRS_INIT_NAME)

    # debugging
    if "__init__" in ctx.cls.info.names:
        symbol_node = ctx.cls.info.names["__init__"]
        print(f"Symbol table __init__: {symbol_node}")
        if hasattr(symbol_node, "node") and hasattr(symbol_node.node, "type"):
            print(f"Symbol table signature: {symbol_node.node.type.arg_names}")

    if hasattr(ctx.cls.info, "__init__"):
        print(f"TypeInfo __init__: {ctx.cls.info.__init__}")

    print(f"TypeInfo names keys: {list(ctx.cls.info.names.keys())}")
    if "__init__" in ctx.cls.info.names:
        init_symbol = ctx.cls.info.names["__init__"]
        if hasattr(init_symbol, "node"):
            func_def = init_symbol.node
            print(f"FuncDef type: {type(func_def)}")
            print(f"FuncDef.type: {func_def.type}")
            print(f"FuncDef id: {id(func_def)}")

    if "__init__" in ctx.cls.info.names:
        print("✓ __init__ found in symbol table")
        init_node = ctx.cls.info.names["__init__"].node
        if hasattr(init_node, "type") and hasattr(init_node.type, "arg_names"):
            expected_extras = ["name", "dims", "parent", "strict", "data"]
            actual_args = init_node.type.arg_names
            for extra in expected_extras:
                if extra in actual_args:
                    print(f"✓ {extra} found in signature")
                else:
                    print(f"✗ {extra} MISSING from signature")
        else:
            print("✗ __init__ has no type or arg_names")
    else:
        print("✗ __init__ NOT found in symbol table")

    if order:
        _add_order(ctx, adder)
    if frozen:
        _make_frozen(ctx, attributes)
        # Frozen classes are hashable by default, even if inheriting from non-frozen ones.
        hashable: bool | None = _get_decorator_bool_argument(
            ctx, "hash", True
        ) and _get_decorator_bool_argument(ctx, "unsafe_hash", True)
    else:
        hashable = _get_decorator_optional_bool_argument(ctx, "unsafe_hash")
        if hashable is None:  # unspecified
            hashable = _get_decorator_optional_bool_argument(ctx, "hash")

    eq = _get_decorator_optional_bool_argument(ctx, "eq")
    has_own_hash = "__hash__" in ctx.cls.info.names

    if has_own_hash or (hashable is None and eq is False):
        pass  # Do nothing.
    elif hashable:
        # We copy the `__hash__` signature from `object` to make them hashable.
        ctx.cls.info.names["__hash__"] = ctx.cls.info.mro[-1].names["__hash__"]
    else:
        _remove_hashability(ctx)

    return True


class XattreePlugin(Plugin):
    def get_class_decorator_hook(self, fullname):
        if fullname in attr_class_makers:
            return xattree_class_maker
        elif fullname in attrs_class_makers:
            return partial(
                attrs_class_maker,
                auto_attribs_default=True,
                frozen_default=False,
                slots_default=False,
            )
        return None


def plugin(version):
    return XattreePlugin
