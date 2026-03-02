import type { Shape, UOP } from "../uops.ts";
import type { RuntimeExec } from "./types.ts";

const coords = (i: number, dims: number[]): number[] => {
  const c = new Array(dims.length);
  for (let d = dims.length - 1; d >= 0; d--) {
    c[d] = i % dims[d];
    i = (i / dims[d]) | 0;
  }
  return c;
};

const valid = (c: number[], s: Shape): boolean => {
  if (!s.mask) return true;
  for (let d = 0; d < c.length; d++) {
    if (c[d] < s.mask[d][0] || c[d] >= s.mask[d][1]) return false;
  }
  return true;
};

const baseIndex = (c: number[], s: Shape): number => {
  let o = s.offset ?? 0;
  for (let d = 0; d < c.length; d++) o += c[d] * s.strides[d];
  return o;
};

const has = (a: number[], v: number): boolean => {
  for (let i = 0; i < a.length; i++) if (a[i] === v) return true;
  return false;
};

const randAt = (j: number, seed: number): number => {
  let x = (j | 0) ^ (seed | 0) ^ 0x9e3779b9;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return (x >>> 0) / 4294967296;
};

const mix = (oi: number, ri: number, os: Shape, ins: Shape, rd: number[]): number => {
  let li = 0;
  let m = 1;
  let od = os.dims.length - 1;
  for (let d = ins.dims.length - 1; d >= 0; d--) {
    let c: number;
    if (has(rd, d)) {
      c = ri % ins.dims[d];
      ri = (ri / ins.dims[d]) | 0;
    } else {
      c = oi % os.dims[od];
      oi = (oi / os.dims[od]) | 0;
      od--;
    }
    li += c * m;
    m *= ins.dims[d];
  }
  return li;
};

const valueAt = (node: UOP, i: number, s: Shape): number => {
  if (node.op === "CONST") {
    const c = coords(i, s.dims);
    if (!valid(c, s)) return 0;
    const len = node.data.length;
    if (len <= 1) return node.data[0] ?? 0;
    const j = ((baseIndex(c, s) % len) + len) % len;
    return node.data[j];
  }
  if (node.op === "RANGE") return i;
  if (node.op === "RAND") {
    const c = coords(i, s.dims);
    if (!valid(c, s)) return 0;
    return randAt(baseIndex(c, s), node.seed);
  }
  if (node.op === "REDUCE") {
    let acc = node.bin === "ADD" ? 0 : 1;
    const rnum = node.dims.map((d) => node.inShape.dims[d]).reduce((a, c) => a * c, 1);
    for (let r = 0; r < rnum; r++) {
      const v = valueAt(node.src, mix(i, r, s, node.inShape, node.dims), node.inShape);
      acc = node.bin === "ADD" ? acc + v : acc * v;
    }
    return acc;
  }
  const a = valueAt(node.srcs[0], i, node.srcShapes[0]);
  const b = valueAt(node.srcs[1], i, node.srcShapes[1]);
  return node.op === "ADD" ? a + b : a * b;
};

export const exec: RuntimeExec = (uop, shape) => {
  const out = new Array(shape.numel);
  for (let i = 0; i < shape.numel; i++) out[i] = valueAt(uop, i, shape);
  return out;
};
