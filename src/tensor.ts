import { exec, type RuntimeName } from "./runtime/index.ts";
import type { Shape, UOP } from "./uops.ts";


type TensorMethods = {
  add: (b: Tensor) => Tensor;
  sum: (dims?: number[]) => Tensor;
  prod: (dims?: number[]) => Tensor;
  reshape: (dims: number[]) => Tensor;
  permute: (axes: number[]) => Tensor;
  expand: (dims: number[]) => Tensor;
  pad: (pads: [number, number][]) => Tensor;
  shrink: (cuts: [number, number][]) => Tensor;
  run: (backend?: RuntimeName) => number[];
};

export type TensorData = {
  uop: UOP;
  shape: Shape;
};
export type Tensor = TensorData & TensorMethods;
export type Raw = number | Raw[];

export const Tensor = {
  const: (value: number, dims: number[]): Tensor => mkTensor({
    uop: { op: "CONST", data: [value] },
    shape: {
      dims,
      strides: dims.map((_, i) => dims.slice(i + 1).reduce((a, c) => a * c, 1)),
      numel: dims.reduce((a, c) => a * c, 1)
    },
  }),
  new: (raw: Raw = 0) => {
    const dims: number[] = [];
    let cur: Raw = raw;
    while (Array.isArray(cur)) {
      dims.push(cur.length);
      cur = cur[0] as Raw;
    }
    return mkTensor({
      uop: {
        op: "CONST",
        data: ([raw] as number[]).flat(Infinity) as number[]
      },
      shape: {
        dims,
        strides: dims.map((_, i) => dims.slice(i + 1).reduce((a, c) => a * c, 1)),
        numel: dims.reduce((a, c) => a * c, 1)
      },
    });
  }
};


export const BACKEND : {default: RuntimeName} = {default: "js"}

const mkTensor = (t: TensorData): Tensor => {
  const reduceShape = (dims: number[]) => {
    const drop = new Set(dims);
    const outDims = t.shape.dims.filter((_, i) => !drop.has(i));
    return {
      dims: outDims,
      strides: outDims.map((_, i) => outDims.slice(i + 1).reduce((a, c) => a * c, 1)),
      numel: outDims.reduce((a, c) => a * c, 1)
    };
  };
  const reduceDims = (dims?: number[]) => {
    const n = t.shape.dims.length;
    const base = (dims && dims.length) ? dims : [...Array(n)].map((_, i) => i);
    return [...new Set(base.map((d) => (d < 0 ? n + d : d)))].sort((a, b) => a - b);
  };
  return {
    ...t,
    add: (b) => mkTensor({
      ...t,
      uop: { op: "ADD", srcs: [t.uop, b.uop], srcShapes: [t.shape, b.shape] }
    }),
    sum: (dims) => mkTensor({
      uop: { op: "REDUCE", bin: "ADD", src: t.uop, inShape: t.shape, dims: reduceDims(dims) },
      shape: reduceShape(reduceDims(dims)),
    }),
    prod: (dims) => mkTensor({
      uop: { op: "REDUCE", bin: "MUL", src: t.uop, inShape: t.shape, dims: reduceDims(dims) },
      shape: reduceShape(reduceDims(dims)),
    }),
    reshape: (dims) => mkTensor({
      ...t,
      shape: {
        dims,
        strides: dims.map((_, i) => dims.slice(i + 1).reduce((a, c) => a * c, 1)),
        numel: dims.reduce((a, c) => a * c, 1)
      }
    }),
    permute: (axes) => mkTensor({
      ...t,
      shape: {
        dims: axes.map((a) => t.shape.dims[a]),
        strides: axes.map((a) => t.shape.strides[a]),
        numel: t.shape.numel
      }
    }),
    expand: (dims) => mkTensor({
      ...t,
      shape: {
        dims,
        strides: dims.map((d, i) => {
          const sd = t.shape.dims[i] ?? 1;
          const ss = t.shape.strides[i] ?? 0;
          if (sd === d) return ss;
          if (sd === 1 && d >= 1) return 0;
          throw new Error("bad expand");
        }),
        numel: dims.reduce((a, c) => a * c, 1)
      }
    }),
    pad: (pads) => mkTensor({
      ...t,
      shape: {
        dims: t.shape.dims.map((d, i) => pads[i][0] + d + pads[i][1]),
        strides: t.shape.strides,
        numel: t.shape.dims
          .map((d, i) => pads[i][0] + d + pads[i][1])
          .reduce((a, c) => a * c, 1),
        offset: (t.shape.offset ?? 0) - pads.reduce((a, p, i) => a + p[0] * t.shape.strides[i], 0),
        mask: t.shape.dims.map((d, i) => [pads[i][0], pads[i][0] + d])
      }
    }),
    shrink: (cuts) => mkTensor({
      ...t,
      shape: {
        dims: cuts.map((c) => c[1] - c[0]),
        strides: t.shape.strides,
        numel: cuts.map((c) => c[1] - c[0]).reduce((a, c) => a * c, 1),
        offset: (t.shape.offset ?? 0) + cuts.reduce((a, c, i) => a + c[0] * t.shape.strides[i], 0)
      }
    }),
    run: (backend?: RuntimeName) => exec(backend ?? BACKEND.default, t.uop, t.shape)
  };
};
