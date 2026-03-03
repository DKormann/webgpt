import { RAWBUFFER, TensorShape, UOp } from "./types";
import { uop } from "./uops";

export type TensorLike = { uop: UOp; shape: TensorShape };

export type Kernelized = {
  graph: UOp;
  buffers: RAWBUFFER[];
  output: RAWBUFFER;
};

const randAt = (j: number, seed: number): number => {
  let x = (j | 0) ^ (seed | 0) ^ 0x9e3779b9;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return (x >>> 0) / 4294967296;
};

const makeBuffer = (data: number[]): RAWBUFFER => {
  const arr = data.slice();
  return {
    size: arr.length,
    read: async () => arr.slice()
  };
};

const materializeLeaf = (uop: UOp, shape: TensorShape): number[] => {
  if (uop.op === "CONST") {
    const out = new Array<number>(shape.numel);
    const data = uop.data ?? [uop.val ?? 0];
    const len = data.length;
    for (let i = 0; i < shape.numel; i++) out[i] = len ? data[((i % len) + len) % len] : 0;
    return out;
  }
  if (uop.op === "RAND") {
    const out = new Array<number>(shape.numel);
    for (let i = 0; i < shape.numel; i++) out[i] = randAt(i, uop.seed);
    return out;
  }
  throw new Error(`kernelize leaf expected CONST/RAND, got ${uop.op}`);
};

const toExpr = (
  node: UOp,
  shape: TensorShape,
  bufs: RAWBUFFER[]
): UOp => {
  if (node.op === "CONST" || node.op === "RAND") {
    const b = makeBuffer(materializeLeaf(node, shape));
    bufs.push(b);
    return uop.view(uop.buffer(b), [{ dims: shape.dims, strides: shape.strides }]);
  }
  if (node.op === "MUL") {
    return uop.mul(toExpr(node.srcs[0], shape, bufs), toExpr(node.srcs[1], shape, bufs));
  }
  throw new Error(`kernelize expression unsupported op: ${node.op}`);
};

export const kernelize = (tensor: TensorLike): Kernelized => {
  const buffers: RAWBUFFER[] = [];
  const out = makeBuffer(new Array<number>(tensor.shape.numel).fill(0));
  buffers.push(out);

  const outView = uop.view(uop.buffer(out), [{ dims: tensor.shape.dims, strides: tensor.shape.strides }]);

  if (tensor.uop.op === "MATMUL") {
    const [aNode, bNode] = tensor.uop.srcs;
    const [aShape, bShape] = tensor.uop.srcShapes;
    if (aShape.dims.length !== 2 || bShape.dims.length !== 2) throw new Error("kernelize matmul expects 2D inputs");
    const [M, K] = aShape.dims;
    const [, N] = bShape.dims;

    const aBuf = makeBuffer(materializeLeaf(aNode, aShape));
    const bBuf = makeBuffer(materializeLeaf(bNode, bShape));
    buffers.push(aBuf, bBuf);

    const aView3 = uop.view(uop.buffer(aBuf), [{ dims: [M, N, K], strides: [K, 0, 1] }]);
    const bView3 = uop.view(uop.buffer(bBuf), [{ dims: [M, N, K], strides: [0, 1, N] }]);
    const graph = uop.store(uop.reduce(uop.mul(aView3, bView3), 2, "ADD"), outView);

    return { graph, buffers, output: out };
  }

  const graph = uop.store(toExpr(tensor.uop, tensor.shape, buffers), outView);
  return { graph, buffers, output: out };
};
