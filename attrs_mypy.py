from mypy.plugin import Plugin
from mypy.plugins.attrs import (
   attr_attrib_makers,
   attr_class_makers,
   attr_dataclass_makers,
)

# These work just like `attr.dataclass`.
attr_dataclass_makers.add("xattree.xattree")

# This works just like `attr.s`.
attr_class_makers.add("xattree.xattree")

# These are our `attr.ib` makers.
attr_attrib_makers.add("xattree.field")
attr_attrib_makers.add("xattree.dim")
attr_attrib_makers.add("xattree.coord")
attr_attrib_makers.add("xattree.array")

class XattreePlugin(Plugin):
    # Our plugin does nothing but it has to exist so this file gets loaded.
    pass


def plugin(version):
    return XattreePlugin