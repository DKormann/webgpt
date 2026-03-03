import { UOp, TensorShape } from "./types";

export type Raw = number | Raw[];
export type RuntimeName = "js" | "webgpu";

export type Tensor = {
  uop: UOp;
  shape: TensorShape;
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

const mkShape = (dims: number[]): TensorShape => ({
  dims,
  strides: dims.map((_, i) => dims.slice(i + 1).reduce((a, c) => a * c, 1)),
  numel: dims.reduce((a, c) => a * c, 1)
});

const flattenRaw = (raw: Raw): number[] => ([raw] as number[]).flat(Infinity) as number[];

const shapeFromRaw = (raw: Raw): TensorShape => {
  const dims: number[] = [];
  let cur: Raw = raw;
  while (Array.isArray(cur)) {
    dims.push(cur.length);
    cur = cur[0] as Raw;
  }
  return mkShape(dims);
};

const mkTensor = (uop: UOp, shape: TensorShape): Tensor => {
  const self = {} as Tensor;
  self.uop = uop;
  self.shape = shape;

  self.mul = (other) => {
    if (self.shape.numel !== other.shape.numel) throw new Error("mul expects same numel");
    return mkTensor({ op: "MUL", srcs: [self.uop, other.uop] }, self.shape);
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
    throw new Error("Tensor.run not implemented yet")
  };

  return self;
};

export const Tensor = {
  const: (value: number, dims: number[]): Tensor =>
    mkTensor({ op: "CONST", srcs: [], val: value }, mkShape(dims)),

  new: (raw: Raw = 0): Tensor =>
    mkTensor({ op: "CONST", srcs: [], data: flattenRaw(raw), val: flattenRaw(raw)[0] ?? 0 }, shapeFromRaw(raw)),

  rand: (dims: number[]): Tensor => mkTensor({ op: "RAND", srcs: [], seed: (Math.random() * 0x7fffffff) | 0 }, mkShape(dims))
};
