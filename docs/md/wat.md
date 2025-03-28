# What tree?

`xattree` ("exa-tree", or "cat tree" if you like) is an [`xarray`](https://xarray.dev/) integration for [`attrs`](https://www.attrs.org/en/stable/), or vice versa.

**Why?**

> [W]e cannot seem to solve the problem of separation of powers. We are not even close. We do not agree on what the principle requires, what its objectives are, or how it does or could accomplish its objectives. &mdash; Elizabeth Magill<sup>[1]</sup>

You are sovereign. Your will is law, filtered though it may be through untold layers of abstraction and indirection. Surveying your domain, you discern disorder. The dimensions are wrong. Unhelpful perspectives proliferate. 

With `xarray` harmony is possible. Coordinating so many views promises to be like herding cats, but provided a [good tree](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html), they'll sort themselves out.

Your domain becomes legible to a powerful new state management apparatus. Like a well-oiled executive or judiciary, `xattree` props up your class hierarchy &mdash; respecting the "letter", i.e. semblance and behavior, while molding the spirit into such shape as to guarantee [inheritances](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html#alignment-and-coordinate-inheritance), etc.

Your constituents, no longer wholly responsible for, or indeed possessed of, their respective properties, fall quickly into line. Tranquility prevails.

Your Janus-faced program pleases your stakeholders and yourself.

**How?**

> [T]he "separation of concerns", which, even if not perfectly possible, is yet the only available technique for effective ordering of one's thoughts, that I know of... is being one- and multiple-track minded simultaneously. &mdash; Edsger Dijkstra<sup>[2]</sup>

In an object model, every element knows its place and, ideally, little else. Encapsulation is the name of the game. For analysis, the opposite is true &mdash; context-awareness underlies expressive power.

One desires both.

Like a homicidal psycho jungle cat, `xattree` claws itself into your `attrs` object model at import time. There it remains like toxoplasmosis until runtime, at which point it consumes the soul (`__dict__`) of unsuspecting instances and substitutes itself (an `xarray.DataTree`).


[1]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=224797

[2]: https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD447.html