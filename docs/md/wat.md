# What tree?

`xattree` ("exa-tree", or "cat tree" if you like) is an [`xarray`](https://xarray.dev/) integration for [`attrs`](https://www.attrs.org/en/stable/), or vice versa.

**Why?**

> [W]e cannot seem to solve the problem of separation of powers. We are not even close. We do not agree on what the principle requires, what its objectives are, or how it does or could accomplish its objectives. &mdash; Elizabeth Magill<sup>[1]</sup>

You are sovereign. Your will is law, filtered though it may be through untold layers of abstraction and indirection. Surveying your domain, you discern disorder. Dimensions threaten to shift. Perspectives proliferate. 

With `xarray` harmony is possible. Coordinating views is like herding cats, but provided a [good tree](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html), they'll sort themselves out.

Your realm becomes legible to a new state management apparatus. Like a well-oiled executive, `xattree` props up your class hierarchy &mdash; respecting the "letter", i.e. semblance and behavior, while molding the spirit so as to guarantee alignment, protect [inheritances](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html#alignment-and-coordinate-inheritance), etc.

Your constituents, no longer wholly responsible for (or indeed possessed of) their respective properties, fall quickly into line. Tranquility prevails. Your Janus-faced program pleases your stakeholders and yourself.

**How?**

> [T]he "separation of concerns", which, even if not perfectly possible, is yet the only available technique for effective ordering of one's thoughts... is being one- and multiple-track minded simultaneously. &mdash; Edsger Dijkstra<sup>[2]</sup>

Like a homicidal psycho jungle cat, `xattree` claws itself into your `attrs` object model at import time. There it remains like toxoplasmosis until runtime, at which point it consumes the soul (`__dict__`) of unsuspecting instances and substitutes itself (an `xarray.DataTree`).

```{mermaid}
sequenceDiagram
    Note over caller: initialize
    caller->>host: __init__()
    Note over host,__dict__: attrs-generated initialization
    host->>__dict__: [populates]
    create participant DataTree
    Note over host,DataTree: xarray data tree initialization
    host->>DataTree: [creates]
    __dict__->>DataTree: [transfers everything to]
    opt 
    host-->>parent: [bind parent]
    DataTree<<-->>parent: [bind data trees,<br/>align dimensions]
    end
    DataTree->>host: [return]
    Note over host,DataTree: (__dict__ now mostly empty,<br/>node's dimensions aligned)
    host->>caller: [return]
    
    Note over caller: get variable
    caller->>host: .x
    Note over host,DataTree: override __getattr__
    alt is array
    host->>DataTree: .data["x"]
    else is dimension
    host->>DataTree: .data.dims["x"]
    else is scalar
    host->>DataTree: .data.attrs["x"]
    end
    DataTree->>host: [return value]
    host->>caller: [return value]
    
    Note over caller: set variable
    caller->>host: .x = ...
    Note over host,DataTree: override __setattr__
    alt is array
    host->>DataTree: .data["x"] = ...
    DataTree->>DataTree: [check dimensions]
    else is scalar
    host->>DataTree: .data.attrs["x"] = ...
    end
    DataTree->>host: [return]
    host->>caller: [return]
```

[1]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=224797

[2]: https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD447.html