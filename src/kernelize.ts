import { Schedule, UOp } from "./types";
import { uop } from "./uops";


const stridesFor = (dims: number[]): number[] =>
  dims.map((_, i) => dims.slice(i + 1).reduce((a, c) => a * c, 1));

const numel = (dims: number[]): number => dims.reduce((a, c) => a * c, 1);

const inferDims = (node: UOp): number[] => {
  if (node.op === "VIEW") return node.views[node.views.length - 1]?.dims ?? [1];
  if (node.op === "RESHAPE" || node.op === "EXPAND" || node.op === "PERMUTE") return node.shape;
  if (node.op === "PAD") {
    const src = inferDims(node.srcs[0]);
    return src.map((d, i) => d + (node.args[i]?.[0] ?? 0) + (node.args[i]?.[1] ?? 0));
  }
  if (node.op === "SHRINK") return node.args.map(([a, b]) => b - a);
  if (node.op === "REDUCE") {
    const src = inferDims(node.srcs[0]);
    if (node.axis < 0 || node.axis >= src.length) throw new Error(`kernelize bad reduce axis ${node.axis}`);
    return src.filter((_, i) => i !== node.axis);
  }
  if (node.op === "ADD" || node.op === "MUL") return inferDims(node.srcs[0]);
  if (node.op === "BUFFER") return [node.buf.size];
  if (node.op === "CONST") return [node.val.length || 1];
  if (node.op === "RAND") {
    if (node.size === undefined) throw new Error("kernelize RAND requires shape context");
    return [node.size];
  }
  if (node.op === "INDEX" || node.op === "RANGE") return [1];
  if (node.op === "STORE" || node.op === "ENDRANGE") throw new Error(`kernelize expects high graph, got ${node.op}`);
  throw new Error(`kernelize unsupported op ${(node as UOp).op}`);
};

const makeBuffer = (size: number) => {
  const arr = new Array<number>(size).fill(0);
  return {
    size,
    read: async () => arr.slice()
  };
};

export const kernelize = (graph: UOp[]) : Schedule => {
  const Buffers = graph.map((g) => makeBuffer(numel(inferDims(g))));
  const roots = graph.map((g, i) => {
    const dims = inferDims(g);
    const out = uop.view(uop.buffer(Buffers[i]), [{ dims, strides: stridesFor(dims) }]);
    return uop.store(g, out);
  });

  return {
    items: [{ Buffers, roots }]
  };
}
