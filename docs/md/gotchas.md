# Gotchas

## Don't use this library

Seriously, don't.

- It straight up lies to you.
- It messes with your `__dict__`.
- Every attribute access goes through a slow, convoluted `__getattr__` hook.
- The error messages are patchy at best.
- Have you even looked at the code?

This was a proof of concept for a prototype. Don't invite the cat in. If you insist on it, you will be bitten. 

## Using `cattrs`? You need a hook factory

If you're using [`cattrs`](https://catt.rs/en/stable/index.html) to un/structure `xattree` classes, you must tell `cattrs` to use `xattree`'s own `asdict` function. Simple as:

```python
converter.register_unstructure_hook_factory(
    xattree.has,
    lambda _: xattree.asdict
)
```

By default, it will use `attrs.asdict`, which doesn't know that `xattree`-managed fields (`name`, `parent`, etc) should not be included.

## Static type checkers can't see `xattree`-managed fields

Your average felid is aloof by constitution &mdash; similarly, this library keeps to itself, though it will not prevent you from exposing *your* (classes') guts in public if you so choose.

Since `xattree`-managed fields are attached via [`field_transformer`](https://www.attrs.org/en/stable/extending.html#automatic-field-transformation-and-modification), *they are not visible to static type checkers*. Static checkers don't run your code, they just look at it. This means you will not see fields like `data` (or whatever else you called your `DataTree` field), `parent`, etc in the initializer signature on hovering over your class, nor will you see an informative type hint on hovering over `xattree`-managed attributes. Intellisense *will* work properly on your own fields.

Despite the above, `xattree` updates your class' `__anotations__` with the `xattree`-managed field information, in case you need it at runtime.