# What?

`xattree` ("exa-tree", or "cat tree" if you like) is an [`xarray`](https://xarray.dev/) integration for [`attrs`](https://www.attrs.org/en/stable/), or vice versa.

`xattree` maps your `attrs` objects onto `xarray`'s data model. Given the former, it gives them back. They act the same, but behind it all is a (network of) `xarray.DataTree` node(s).

You get

- a `DataTree` for each instance living in `.data` by default
- dimension and coordinate inheritance
- hierarchical addressing

..and the goodies offered by `xarray` more generally.

**Why?**

> [W]e cannot seem to solve the problem of separation of powers. We are not even close. We do not agree on what the principle requires, what its objectives are, or how it does or could accomplish its objectives. &mdash; Elizabeth Magill<sup>[1]</sup>

You are sovereign. Your will is law, filtered though it may be through untold layers of abstraction and indirection. Surveying your domain, you discern disorder. Dimensions threaten to shift. Perspectives proliferate. 

With `xarray` harmony is possible. Coordinating views is like herding cats, but provided a [good tree](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html), they'll sort themselves out.

Your realm becomes legible to a new state management apparatus. Like a well-oiled executive, `xattree` props up your class hierarchy &mdash; respecting the "letter", i.e. semblance and behavior, while molding the spirit so as to guarantee alignment, protect [inheritances](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html#alignment-and-coordinate-inheritance), etc.

Your constituents, no longer wholly responsible for (or indeed possessed of) their respective properties, fall quickly into line. Tranquility prevails. Your Janus-faced program pleases your stakeholders and yourself.


[1]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=224797
