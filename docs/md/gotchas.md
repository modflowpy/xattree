# Gotchas

## Intellisense

Your average felid is aloof by constitution &mdash; similarly, this library keeps to itself, though it will not prevent you from exposing *your* (classes') guts in public if you so choose.

Since `xattree`-managed fields are attached via [`field_transformer`](https://www.attrs.org/en/stable/extending.html#automatic-field-transformation-and-modification), *they are not visible to static type checkers*. Static checkers don't run your code, they just look at it. This means you will not see fields like `data` (or whatever else you called your `DataTree` field), `parent`, etc in the initializer signature on hovering over your class, nor will you see an informative type hint on hovering over `xattree`-managed attributes. Intellisense *will* work properly on your own fields.

## `__annotations__`

Despite the above, `xattree` updates your class' `__anotations__` with the `xattree`-managed field information, in case you need it at runtime.