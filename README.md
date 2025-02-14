# xattree

`attrs` + `xarray.DataTree` = `xattree`

"x-a-tree", or "cat tree" if you like.

```python
@xattree
@define(slots=False)
class Grid:
    rows: int = field(
        default=3,
        metadata={
            "dim": {
                "coord": "j",
                "scope": "root",
            }
        },
    )
    cols: int = field(
        default=3,
        metadata={
            "dim": {
                "coord": "i",
                "scope": "root",
            },
        },
    )


@xattree
@define(slots=False)
class Arrs:
    arr: NDArray[np.float64] = field(
        default=0.0, metadata={"dims": ("rows", "cols")}
    )


@xattree
@define(slots=False)
class Root:
    grid: Grid = field(default=Factory(Grid), metadata={"bind": True})
    arrs: Arrs = field(default=Factory(Arrs), metadata={"bind": True})


grid = Grid()
root = Root(grid=grid)
arrs = Arrs(root)
root.data
<xarray.DataTree 'root'>
Group: /
│   Dimensions:  (rows: 3, cols: 3)
│   Coordinates:
│       j        (rows) int64 24B 0 1 2
│       i        (cols) int64 24B 0 1 2
│   Dimensions without coordinates: rows, cols
├── Group: /grid
│       Dimensions:  (rows: 3, cols: 3)
│       Coordinates:
│           j        (rows) int64 24B 0 1 2
│           i        (cols) int64 24B 0 1 2
│       Dimensions without coordinates: rows, cols
│       Attributes:
│           rows:     3
│           cols:     3
└── Group: /arrs
        Dimensions:  (rows: 3, cols: 3)
        Coordinates:
            j        (rows) int64 24B 0 1 2
            i        (cols) int64 24B 0 1 2
        Dimensions without coordinates: rows, cols
        Data variables:
            arr      (rows, cols) float64 72B 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
```