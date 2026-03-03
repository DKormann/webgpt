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



## LAYOUT

1. Tensor: high level frontend for Tensor data


2. Kernelize: break up Tensor into what later becomes kernels. each one is a set of stores.


3. linearize: create low level graph. translate all views into ranges and indexes.


4. backend: generate shader code to run the kernels

