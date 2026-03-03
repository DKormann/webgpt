export type Raw = number | Raw[];
export type RuntimeName = "js" | "webgpu";

export type Shape = {
  dims: number[];
  strides: number[];
  numel: number;
  offset?: number;
  mask?: [number, number][];
};


export type Tensor = {
  uop: UOp;
  shape: Shape;
  mul: (other: Tensor) => Tensor;
  matmul: (other: Tensor) => Tensor;
  reshape: (dims: number[]) => Tensor;
  permute: (axes: number[]) => Tensor;
  expand: (dims: number[]) => Tensor;
  pad: (pads: [number, number][]) => Tensor;
  shrink: (cuts: [number, number][]) => Tensor;
  run: (_backend?: RuntimeName) => Promise<Raw>;
};

export const BACKEND: { default?: RuntimeName } = {};
import { Tensor } from "webgpu-torch";
import { kernelize } from "./kernelize";
import { linearize } from "./linearize";

const mkShape = (dims: number[]): Shape => ({
  dims,
  strides: dims.map((_, i) => dims.slice(i + 1).reduce((a, c) => a * c, 1)),
  numel: dims.reduce((a, c) => a * c, 1)
});

const flattenRaw = (raw: Raw): number[] => ([raw] as number[]).flat(Infinity) as number[];

const shapeFromRaw = (raw: Raw): Shape => {
  const dims: number[] = [];
  let cur: Raw = raw;
  while (Array.isArray(cur)) {
    dims.push(cur.length);
    cur = cur[0] as Raw;
  }
  return mkShape(dims);
};

const nest = (flat: number[], dims: number[]): Raw => {
  if (dims.length === 0) return flat[0] ?? 0;
  if (dims.length === 1) return flat.slice(0, dims[0]);
  const step = dims.slice(1).reduce((a, c) => a * c, 1);
  const out: Raw[] = [];
  for (let i = 0; i < dims[0]; i++) out.push(nest(flat.slice(i * step, (i + 1) * step), dims.slice(1)));
  return out;
};

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

const randAt = (j: number, seed: number): number => {
  let x = (j | 0) ^ (seed | 0) ^ 0x9e3779b9;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return (x >>> 0) / 4294967296;
};

const valueAt = (node: UOp, i: number, s: Shape): number => {
  if (node.op === "CONST") {
    const c = coords(i, s.dims);
    if (!valid(c, s)) return 0;
    const len = node.data.length;
    if (len <= 1) return node.data[0] ?? 0;
    const j = ((baseIndex(c, s) % len) + len) % len;
    return node.data[j];
  }

  if (node.op === "RAND") {
    const c = coords(i, s.dims);
    if (!valid(c, s)) return 0;
    return randAt(baseIndex(c, s), node.seed);
  }

  if (node.op === "MATMUL") {
    const aShape = node.srcShapes[0];
    const bShape = node.srcShapes[1];
    const [m, k] = aShape.dims;
    const [, n] = bShape.dims;
    const row = (i / n) | 0;
    const col = i % n;
    let acc = 0;
    for (let kk = 0; kk < k; kk++) {
      const av = valueAt(node.srcs[0], row * k + kk, aShape);
      const bv = valueAt(node.srcs[1], kk * n + col, bShape);
      acc += av * bv;
    }
    return acc;
  }

  const a = valueAt(node.srcs[0], i, node.srcShapes[0]);
  const b = valueAt(node.srcs[1], i, node.srcShapes[1]);
  return a * b;
};

const mkTensor = (uop: UOp, shape: Shape): Tensor => {
  const self = {} as Tensor;
  self.uop = uop;
  self.shape = shape;

  self.mul = (other) => {
    if (self.shape.numel !== other.shape.numel) throw new Error("mul expects same numel");
    return mkTensor(
      { op: "MUL", srcs: [self.uop, other.uop], srcShapes: [self.shape, other.shape] },
      self.shape
    );
  };

  self.matmul = (other) => {
    if (self.shape.dims.length !== 2 || other.shape.dims.length !== 2) {
      throw new Error("matmul expects 2D tensors");
    }
    const [m, k] = self.shape.dims;
    const [k2, n] = other.shape.dims;
    if (k !== k2) throw new Error(`matmul shape mismatch: [${m},${k}] x [${k2},${n}]`);
    return mkTensor(
      { op: "MATMUL", srcs: [self.uop, other.uop], srcShapes: [self.shape, other.shape] },
      mkShape([m, n])
    );
  };

  self.reshape = (dims) => mkTensor(self.uop, mkShape(dims));

  self.permute = (axes) =>
    mkTensor(self.uop, {
      dims: axes.map((a) => self.shape.dims[a]),
      strides: axes.map((a) => self.shape.strides[a]),
      numel: self.shape.numel,
      offset: self.shape.offset,
      mask: self.shape.mask ? axes.map((a) => self.shape.mask![a]) : undefined
    });

  self.expand = (dims) =>
    mkTensor(self.uop, {
      dims,
      strides: dims.map((d, i) => {
        const sd = self.shape.dims[i] ?? 1;
        const ss = self.shape.strides[i] ?? 0;
        if (sd === d) return ss;
        if (sd === 1 && d >= 1) return 0;
        throw new Error("bad expand");
      }),
      numel: dims.reduce((a, c) => a * c, 1),
      offset: self.shape.offset,
      mask: self.shape.mask
    });

  self.pad = (pads) =>
    mkTensor(self.uop, {
      dims: self.shape.dims.map((d, i) => pads[i][0] + d + pads[i][1]),
      strides: self.shape.strides,
      numel: self.shape.dims.map((d, i) => pads[i][0] + d + pads[i][1]).reduce((a, c) => a * c, 1),
      offset: (self.shape.offset ?? 0) - pads.reduce((a, p, i) => a + p[0] * self.shape.strides[i], 0),
      mask: self.shape.dims.map((d, i) => [pads[i][0], pads[i][0] + d])
    });

  self.shrink = (cuts) =>
    mkTensor(self.uop, {
      dims: cuts.map((c) => c[1] - c[0]),
      strides: self.shape.strides,
      numel: cuts.map((c) => c[1] - c[0]).reduce((a, c) => a * c, 1),
      offset: (self.shape.offset ?? 0) + cuts.reduce((a, c, i) => a + c[0] * self.shape.strides[i], 0)
    });

  self.run = async (_backend?: RuntimeName) => {
    // Pipeline shape: tensor graph -> kernelize -> linearize.
    const k = kernelize(self);
    linearize(k.graph, [k.output]);

    const out = new Array<number>(self.shape.numel);
    for (let i = 0; i < self.shape.numel; i++) out[i] = valueAt(self.uop, i, self.shape);
    return nest(out, self.shape.dims);
  };

  return self;
};

export const Tensor = {
  new: (raw: Raw = 0): Tensor => mkTensor({ op: "CONST", data: flattenRaw(raw) }, shapeFromRaw(raw)),

  rand: (dims: number[]): Tensor => mkTensor({ op: "RAND", seed: (Math.random() * 0x7fffffff) | 0 }, mkShape(dims))
};
