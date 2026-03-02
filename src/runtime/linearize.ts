import type { LinearMatmul, UOP } from "../uops.ts";

export const linearize = (uop: UOP): LinearMatmul | null => {
  if (uop.op !== "REDUCE" || uop.bin !== "ADD" || uop.dims.length !== 1 || uop.dims[0] !== 1) return null;
  if (uop.src.op !== "MUL") return null;
  const [a, b] = uop.src.srcs;
  const [as, bs] = uop.src.srcShapes;
  if (a.op !== "CONST" || b.op !== "CONST") return null;
  if (as.dims.length !== 3 || bs.dims.length !== 3) return null;
  const [M, K, N] = as.dims;
  if (bs.dims[0] !== M || bs.dims[1] !== K || bs.dims[2] !== N) return null;
  // Matmul broadcasting pattern from tensor.matmul:
  // A: [M,K,1] expand([M,K,N]) => strides [K,1,0]
  // B: [1,K,N] expand([M,K,N]) => strides [0,N,1]
  if (as.strides[2] !== 0 || bs.strides[0] !== 0 || bs.strides[2] !== 1) return null;

  const t = 16;
  return {
    kind: "matmul",
    M,
    N,
    K,
    tile: [t, t, t],
    workgroup: [t, t, 1],
    a: { data: a.data, shape: as },
    b: { data: b.data, shape: bs },
    ops: [
      { op: "RANGE", id: "wg_y", kind: "global", size: Math.ceil(M / t) },
      { op: "RANGE", id: "wg_x", kind: "global", size: Math.ceil(N / t) },
      { op: "RANGE", id: "ly", kind: "local", size: t },
      { op: "RANGE", id: "lx", kind: "local", size: t },
      { op: "DEFINE_LOCAL", id: "As", shape: [t, t] },
      { op: "DEFINE_LOCAL", id: "Bs", shape: [t, t] },
      { op: "RANGE", id: "k0", kind: "reduce", size: Math.ceil(K / t) },
      { op: "LOAD", id: "As", from: "A", scope: "global" },
      { op: "LOAD", id: "Bs", from: "B", scope: "global" },
      { op: "BARRIER" },
      { op: "RANGE", id: "k1", kind: "reduce", size: t },
      { op: "MULACC", a: "As", b: "Bs", acc: "acc" },
      { op: "ENDRANGE", id: "k1" },
      { op: "BARRIER" },
      { op: "ENDRANGE", id: "k0" },
      { op: "STORE", to: "C" },
      { op: "ENDRANGE", id: "lx" },
      { op: "ENDRANGE", id: "ly" },
      { op: "ENDRANGE", id: "wg_x" },
      { op: "ENDRANGE", id: "wg_y" }
    ]
  };
};
