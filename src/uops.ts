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
      op: "RAND";
      seed: number;
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

const uopKids = (u: UOP): UOP[] => (u.op === "REDUCE" ? [u.src] : u.op === "ADD" || u.op === "MUL" ? [u.srcs[0], u.srcs[1]] : []);

const uopLine = (u: UOP): string => {
  if (u.op === "CONST") return `CONST len=${u.data.length}`;
  if (u.op === "RANGE") return `RANGE count=${u.count}`;
  if (u.op === "RAND") return `RAND seed=${u.seed}`;
  if (u.op === "REDUCE") return `REDUCE ${u.bin} dims=[${u.dims.join(",")}] in=[${u.inShape.dims.join("x")}]`;
  return `${u.op} a=[${u.srcShapes[0].dims.join("x")}] b=[${u.srcShapes[1].dims.join("x")}]`;
};

export const formatUOP = (root: UOP): string => {
  const ids = new Map<UOP, string>();
  const visit = (u: UOP) => {
    if (ids.has(u)) return;
    ids.set(u, `n${ids.size}`);
    for (const k of uopKids(u)) visit(k);
  };
  visit(root);

  const emitted = new Set<UOP>();
  const lines: string[] = [];
  const walk = (u: UOP, prefix: string, last: boolean) => {
    const id = ids.get(u) ?? "nx";
    const head = `${prefix}${prefix ? (last ? "`- " : "|- ") : ""}${id} ${uopLine(u)}`;
    if (emitted.has(u)) {
      lines.push(`${head} (ref)`);
      return;
    }
    lines.push(head);
    emitted.add(u);
    const kids = uopKids(u);
    for (let i = 0; i < kids.length; i++) walk(kids[i], `${prefix}${prefix ? (last ? "   " : "|  ") : ""}`, i === kids.length - 1);
  };
  walk(root, "", true);
  return lines.join("\n");
};

export type UOPGraphItem = {
  id: string;
  op: UOP["op"];
  inputs: string;
  label: string;
};

export const graphUOP = (root: UOP): UOPGraphItem[] => {
  const ids = new Map<UOP, string>();
  const seen = new Set<UOP>();
  const order: UOP[] = [];
  const visit = (u: UOP) => {
    if (seen.has(u)) return;
    seen.add(u);
    if (!ids.has(u)) ids.set(u, `n${ids.size}`);
    for (const k of uopKids(u)) {
      if (!ids.has(k)) ids.set(k, `n${ids.size}`);
      visit(k);
    }
    order.push(u);
  };
  visit(root);
  return order.map((u) => ({
    id: ids.get(u) ?? "nx",
    op: u.op,
    inputs: uopKids(u).map((k) => ids.get(k) ?? "nx").join(","),
    label: uopLine(u)
  }));
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
