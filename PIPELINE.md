
Project goal: minimal Tensor library

Build lazy graph
Tensor ops create a UOp DAG (including movement/view metadata), no immediate compute.

Kernelize
Rewrite DAG into Ops.KERNEL + Ops.ASSIGN, fuse where legal, insert graph barriers (GBARRIER), normalize views, build per-kernel ASTs.
Entry: get_kernelize_map.

Schedule
Turn kernelized DAG into ordered ScheduleItems using assign dependencies; bind symbolic vars; memory-plan/reuse buffers.
Entry: create_schedule_with_vars.

Lower + optimize program
For each kernel AST: apply kernel opts (heuristics/beam), lower views/shapes to explicit index math (INDEX/valid), linearize UOps.
Entries: get_optimized_ast, full_rewrite.

Render source
Backend renderer converts linear UOps to code (C/OpenCL/PTX/WGSL/etc.) with explicit loads/stores and launch dims.
See ProgramSpec.

Compile + run
CompiledRunner compiles/caches source, creates runtime program, allocates buffers as needed, launches kernels in schedule order.
Entry: run_schedule.