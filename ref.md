# tinygrad reference

Tensor graph -> Kernelize/Schedule -> Lower/Rewrites -> Linearize -> Render/Runtime

Main responsibilities on the UOp graph:

1. Tensor / lazy graph build
  Builds high-level UOp DAG (math + views + reduce semantics).
  Keeps shape logic in VIEW/ShapeTracker, not explicit index math yet.
2. kernelize + schedule
  Decides materialization boundaries (what becomes a kernel output).
  Forms kernel roots (SINK with STORE outputs) and execution ordering/deps.
3. lowerer (early codegen rewrites)
  Converts shape/view semantics into explicit index logic.
  Rewrites LOAD/STORE view-style access into INDEX (+ VALID/gates).
  Rewrites REDUCE_AXIS into lowered REDUCE form.
4. mid/late rewrites (expander/devectorizer/reduce removal/gpudims/final rewrite)
  Expands vector/contract forms, resolves reductions to explicit loop+accumulator structure, inserts backend-specific specials, simplifies algebra.
  Goal: backend-friendly primitive UOps.
5. linearizer
  Converts DAG-like UOps into ordered block/loop structure, then finalized linear UOp list (... RANGE ... ENDRANGE ... SINK).
  Ensures control flow and ordering are explicit for renderers.
6. renderer + runtime
  Renderer maps linear UOps to backend source/ISA (C/Metal/WGSL/PTX/etc).
  Runtime compiles, binds buffers/vars, and launches kernels.

