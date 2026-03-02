minimal web port of nanogpt

we want our code as short as possible.

## UOP-first design

Codegen should stay dumb and mostly mechanical.
All smartness should live in the UOP graph:
- tiling decisions
- workgroup sizes
- memory layout / view transforms
- scheduling and fusion boundaries

Goal: backend codegen stays platform-independent because it only lowers explicit UOP intent.

Short-term exception:
- `matmul` may emit a specific hardcoded optimized UOP subgraph for now.
