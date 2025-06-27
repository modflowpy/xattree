# How?

> [T]he "separation of concerns", which, even if not perfectly possible, is yet the only available technique for effective ordering of one's thoughts... is being one- and multiple-track minded simultaneously. &mdash; Edsger Dijkstra<sup>[1]</sup>

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

[1]: https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD447.html