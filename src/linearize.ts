import type { UOp, UOpKind } from "./types";
import { uop } from "./uops";

type Range = UOpKind<"RANGE">;

const uniqRanges = (arr: Range[]): Range[] => {
  const seen = new Set<UOp>();
  const out: Range[] = [];
  for (const r of arr) {
    if (!seen.has(r)) {
      seen.add(r);
      out.push(r);
    }
  }
  return out;
};

const collectRanges = (node: UOp): Range[] => {
  if (node.op === "RANGE") return [node];
  if (node.op === "KERNEL") return collectRanges(node.srcs[0]);
  if (node.op === "ADD" || node.op === "MUL" || node.op === "INDEX") {
    return uniqRanges(node.srcs.flatMap((s) => collectRanges(s)) as Range[]);
  }
  if (node.op === "REDUCE") return collectRanges(node.srcs[0]);
  return [];
};

const normalizeExpr = (node: UOp): UOp => {
  if (node.op === "KERNEL") return normalizeExpr(node.srcs[0]);
  if (node.op === "INDEX") {
    const base = normalizeExpr(node.srcs[0]);
    const idx = normalizeExpr(node.srcs[1]);
    if (base.op === "KERNEL") return uop.index(normalizeExpr(base.srcs[0]), idx);
    if (base.op === "INDEX") return uop.index(normalizeExpr(base.srcs[0]), idx);
    return uop.index(base, idx);
  }
  if (node.op === "ADD" || node.op === "MUL") {
    return { op: node.op, srcs: [normalizeExpr(node.srcs[0]), normalizeExpr(node.srcs[1])] };
  }
  if (node.op === "REDUCE") {
    return { op: "REDUCE", bin: node.bin, keep: node.keep, srcs: [normalizeExpr(node.srcs[0])] };
  }
  return node;
};

const flattenRanges = (ranges: Range[]): UOp => {
  if (ranges.length === 0) return uop.const(0);
  let idx: UOp = uop.const(0);
  for (let i = 0; i < ranges.length; i++) {
    const stride = ranges.slice(i + 1).reduce((a, r) => a * r.max, 1);
    const term = stride === 1 ? ranges[i] : uop.mul(ranges[i], uop.const(stride));
    idx = i === 0 ? term : uop.add(idx, term);
  }
  return idx;
};

const asDstIndex = (dst: UOp, outputRanges: Range[]): UOp & { op: "INDEX" } => {
  if (dst.op === "INDEX") return dst;
  if (dst.op === "BUFFER") return uop.index(dst, flattenRanges(outputRanges)) as UOp & { op: "INDEX" };
  throw new Error(`linearize unsupported store dst: ${dst.op}`);
};

const linearizeStore = (st: UOp & { op: "STORE" }): UOp[] => {
  const src = st.srcs[0];
  const dst = st.srcs[1];

  if (src.op === "REDUCE") {
    if (src.bin !== "ADD") throw new Error("linearize only supports REDUCE ADD");

    const outputRanges = uniqRanges(src.keep);
    const redExpr = src.srcs[0];
    const allRanges = collectRanges(redExpr);
    const redRanges = allRanges.filter((r) => !outputRanges.includes(r));

    const d = asDstIndex(dst, outputRanges);
    if (d.srcs[0].op !== "BUFFER") throw new Error("linearize reduce dst must index a BUFFER");

    return [
      ...outputRanges,
      uop.store(uop.const(0), d),
      ...redRanges,
      uop.store(uop.add(uop.index(d.srcs[0], d.srcs[1]), redExpr), d),
      ...redRanges.slice().reverse().map((r) => uop.endrange(r)),
      ...outputRanges.slice().reverse().map((r) => uop.endrange(r)),
    ];
  }

  const outputRanges = collectRanges(src);
  const d = asDstIndex(dst, outputRanges);
  return [
    ...outputRanges,
    uop.store(src, d),
    ...outputRanges.slice().reverse().map((r) => uop.endrange(r)),
  ];
};

export const linearize = (graph: UOp, outBuffer?: UOp & { op: "BUFFER" }): UOp[] => {
  if (graph.op === "KERNEL") {
    const src = normalizeExpr(graph.srcs[0]);
    const out = outBuffer ?? uop.buffer({ size: graph.size, read: async () => [] });
    return linearizeStore(uop.store(src, out));
  }
  if (graph.op === "STORE") return linearizeStore(graph);
  return [normalizeExpr(graph)];
};
