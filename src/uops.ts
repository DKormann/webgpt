export type Shape = {
  dims: number[];
  strides: number[];
  numel: number;
  offset?: number;
  mask?: [number, number][];
};

export type Binop = "ADD" | "MUL";

export type OP = Binop | "RANGE" | "CONST" | "IDX" | "REDUCE";

export type UOP =
  | {
      op: "CONST";
      data: number[];
    }
  | {
      op: "RANGE";
      count: number
    }
  | {
      op: "REDUCE";
      bin: Binop,
      src: UOP,
      inShape: Shape,
      dims: number[]
    }
  | {
      op: Binop;
      srcs: [UOP, UOP];
      srcShapes: [Shape, Shape];
    };

export type KernelUOP =
  | { op: "RANGE"; id: string; kind: "global" | "local" | "reduce"; size: number }
  | { op: "ENDRANGE"; id: string }
  | { op: "DEFINE_LOCAL"; id: string; shape: [number, number] }
  | { op: "LOAD"; id: string; from: "A" | "B"; scope: "global" | "local" }
  | { op: "STORE"; to: "C" }
  | { op: "MULACC"; a: string; b: string; acc: string }
  | { op: "BARRIER" };

export type LinearMatmul = {
  kind: "matmul";
  M: number;
  N: number;
  K: number;
  tile: [number, number, number];
  workgroup: [number, number, number];
  ops: KernelUOP[];
  a: { data: number[]; shape: Shape };
  b: { data: number[]; shape: Shape };
};
