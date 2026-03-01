import { exec } from "./runtime.ts";
import type { Shape, UOP } from "./uops.ts";

type TensorMethods = {
  add: (b: Tensor) => Tensor;
  reshape: (dims: number[]) => Tensor;
  permute: (axes: number[]) => Tensor;
  expand: (dims: number[]) => Tensor;
  run: () => number[];
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
    }
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
      }
    });
  }
};

export const newTensor = (value: number, dims: number[]): Tensor =>
  Tensor.const(value, dims);

const mkTensor = (t: TensorData): Tensor => {
  return {
    ...t,
    add: (b) => mkTensor({
      ...t,
      uop: { op: "ADD", srcs: [t.uop, b.uop] }
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
    run: () => exec(t.uop, t.shape)
  };
};
