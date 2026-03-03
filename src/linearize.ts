import { RAWBUFFER, UOp, View } from "./types";
import { uop } from "./uops";

const isView = (x: UOp): x is UOp & { op: "VIEW"; srcs: [UOp]; views: View[] } => x.op === "VIEW";
const isBuffer = (x: UOp): x is UOp & { op: "BUFFER" } => x.op === "BUFFER";

const resolveBuffer = (x: UOp): UOp & { op: "BUFFER" } => {
  if (isBuffer(x)) return x;
  if (isView(x)) return resolveBuffer(x.srcs[0]);
  throw new Error(`linearize expected BUFFER/VIEW base, got ${x.op}`);
};

const indexFromView = (view: View, ranges: UOp[]): UOp => {
  const n = view.dims.length;
  const use = ranges.slice(Math.max(0, ranges.length - n));

  let idx: UOp = uop.const(0);
  for (let i = 0; i < n; i++) {
    const stride = view.strides[i] ?? 0;
    const r = use[i] ?? uop.const(0);
    const term = stride === 1 ? r : uop.mul(r, uop.const(stride));
    idx = i === 0 ? term : uop.add(idx, term);
  }
  return idx;
};

const lowerExpr = (node: UOp, loops: UOp[]): UOp => {
  if (isView(node)) {
    const view = node.views[node.views.length - 1];
    const base = resolveBuffer(node.srcs[0]);
    return uop.index(base, indexFromView(view, loops));
  }

  if (node.op === "ADD" || node.op === "MUL") {
    return {
      op: node.op,
      srcs: [lowerExpr(node.srcs[0], loops), lowerExpr(node.srcs[1], loops)]
    } as UOp;
  }

  if (node.op === "INDEX") {
    return uop.index(lowerExpr(node.srcs[0], loops), lowerExpr(node.srcs[1], loops));
  }

  return node;
};

const linearizeStore = (graph: UOp & { op: "STORE" }): UOp[] => {
  const src = graph.srcs[0];
  const dst = graph.srcs[1];

  if (src.op === "REDUCE" && src.bin === "ADD") {
    const reducedSrc = src.srcs[0];
    const outBase = isView(dst) ? resolveBuffer(dst.srcs[0]) : isBuffer(dst) ? dst : resolveBuffer(dst.srcs[0]!);
    const outIdx = isView(dst)
      ? indexFromView(dst.views[dst.views.length - 1], [])
      : dst.op === "INDEX"
        ? lowerExpr(dst.srcs[1], [])
        : uop.const(0);

    const max = isView(reducedSrc) ? reducedSrc.views[reducedSrc.views.length - 1].dims[src.axis] : outBase.buf.size;
    const r = uop.range(max);
    const term = lowerExpr(reducedSrc, [r]);

    return [
      uop.store(uop.const(0), uop.index(outBase, outIdx)),
      r,
      uop.store(uop.add(uop.index(outBase, outIdx), term), uop.index(outBase, outIdx)),
      uop.endrange(r)
    ];
  }

  if (isView(dst)) {
    const view = dst.views[dst.views.length - 1];
    const ranges = view.dims.map((d) => uop.range(d));
    const base = resolveBuffer(dst.srcs[0]);
    const store = uop.store(lowerExpr(src, ranges), uop.index(base, indexFromView(view, ranges)));
    return [...ranges, store, ...ranges.slice().reverse().map((r) => uop.endrange(r as UOp & { op: "RANGE" }))];
  }

  if (isBuffer(dst)) {
    return [uop.store(lowerExpr(src, []), uop.index(dst, uop.const(0)))];
  }

  if (dst.op === "INDEX") {
    return [uop.store(lowerExpr(src, []), lowerExpr(dst, []))];
  }

  throw new Error(`linearize STORE destination unsupported: ${dst.op}`);
};

const linearizeReduce = (graph: UOp & { op: "REDUCE" }, buffs?: RAWBUFFER[]): UOp[] => {
  if (graph.bin !== "ADD") throw new Error(`linearize only supports ADD reduce for now`);

  const src = graph.srcs[0];
  const base = resolveBuffer(src);
  const out = uop.buffer((buffs?.[0] ?? base.buf) as RAWBUFFER);
  const zero = uop.const(0);

  const max = isView(src) ? src.views[src.views.length - 1].dims[graph.axis] : out.buf.size;
  const r = uop.range(max);
  const rhs = lowerExpr(src, [r]);

  return [
    uop.store(zero, uop.index(out, zero)),
    r,
    uop.store(uop.add(uop.index(out, zero), rhs), uop.index(out, zero)),
    uop.endrange(r)
  ];
};

export const linearize = (graph: UOp, buffs?: RAWBUFFER[]): UOp[] => {
  if (graph.op === "STORE") return linearizeStore(graph as UOp & { op: "STORE" });
  if (graph.op === "REDUCE") return linearizeReduce(graph as UOp & { op: "REDUCE" }, buffs);
  return [lowerExpr(graph, [])];
};
