import type { BinOp, UOp, UOpKind } from "./types";
import { uop } from "./uops";
import { kernelize } from "./kernelize";
import { linearize } from "./linearize";
import { lowerer } from "./lowerer";
import { WEBGPU } from "./webgpu";
import { DEBUG } from "./debug";
import { numel, stridesFor } from "./helpers";

export type Raw = number | Raw[];
export type RuntimeName = "js" | "webgpu";

export type Tensor = {
  uop: UOp;
  shape: number[];
  numel: () => number;
  mul: (other: Tensor) => Tensor;
  add: (other: Tensor) => Tensor;
  sum: (dims?: number[]) => Tensor;
  matmul: (other: Tensor) => Tensor;
  reshape: (dims: number[]) => Tensor;
  permute: (axes: number[]) => Tensor;
  expand: (dims: number[]) => Tensor;
  pad: (pads: [number, number][]) => Tensor;
  shrink: (cuts: [number, number][]) => Tensor;
  run: (_backend?: RuntimeName) => Promise<Raw>;
};

export const BACKEND: { default: RuntimeName } = { default: "webgpu" };


const flattenRaw = (raw: Raw): number[] => ([raw] as number[]).flat(Infinity) as number[];

const shapeFromRaw = (raw: Raw): number[] => {
  const dims: number[] = [];
  let cur: Raw = raw;
  while (Array.isArray(cur)) {
    dims.push(cur.length);
    cur = cur[0] as Raw;
  }
  return dims;
};

const normalizeAxes = (axes: number[] | undefined, rank: number): number[] => {
  const raw = axes ?? [...Array(rank).keys()];
  const norm = raw.map((a) => (a < 0 ? rank + a : a));
  for (const a of norm) {
    if (a < 0 || a >= rank) throw new Error(`sum axis out of range: ${a} for rank ${rank}`);
  }
  return [...new Set(norm)].sort((a, b) => b - a);
};

const bufferRefsIn = (graph: UOp[]): (UOp & { op: "BUFFER" })[] => {
  const out: (UOp & { op: "BUFFER" })[] = [];
  const seen = new Set<UOp>();
  const walk = (u: UOp) => {
    if (u.op === "BUFFER" && !seen.has(u)) {
      seen.add(u);
      out.push(u);
    }
    u.srcs.forEach(walk);
  };
  graph.forEach(walk);
  return out;
};

const outputBufferIn = (graph: UOp[]): (UOp & { op: "BUFFER" }) => {
  const store = [...graph].reverse().find((u) => u.op === "STORE");
  if (!store || store.op !== "STORE") throw new Error("kernel has no STORE");
  const dst = store.srcs[1];
  if (dst.op !== "INDEX") throw new Error("STORE dst must be INDEX");
  const base = dst.srcs[0];
  if (base.op !== "BUFFER") throw new Error("STORE dst base must be BUFFER");
  return base;
};

const binary = (self: Tensor, op: BinOp) => (other: Tensor) => {
  if (JSON.stringify(self.shape) !== JSON.stringify(other.shape)) throw new Error("shape mismatch");
  return mkTensor({ op, srcs: [self.uop, other.uop] }, self.shape.slice());
};

const reduce = (self: Tensor, op: BinOp, axes?: number[]): Tensor => {
  if (axes === undefined) {
    return mkTensor(uop.reduce(uop.view(self.uop, [{ dims: [self.numel()], strides: [1] }]), [0], op), []);
  }

  const norm = normalizeAxes(axes, self.shape.length);
  if (norm.length === 0) return self;

  let g = self.uop;
  let outShape = self.shape.slice();
  for (const axis of norm) {
    g = uop.reduce(g, [axis], op);
    outShape = outShape.filter((_, i) => i !== axis);
  }
  return mkTensor(g, outShape);
};

const shapeOp = (
  self: Tensor,
  op: "RESHAPE" | "EXPAND" | "PERMUTE" | "PAD" | "SHRINK",
  outShape: number[],
  payload: { shape?: number[]; args?: [number, number][] }
): Tensor => {
  if (op === "PAD" || op === "SHRINK") return mkTensor({ op, srcs: [self.uop], args: payload.args! }, outShape);
  return mkTensor({ op, srcs: [self.uop], shape: payload.shape! }, outShape);
};

const mkTensor = (graph: UOp, shape: number[]): Tensor => {
  const self = {} as Tensor;
  self.uop = graph;
  self.shape = shape;
  self.numel = () => numel(self.shape);

  self.add = binary(self, "ADD");
  self.mul = binary(self, "MUL");
  self.sum = (axes?) => reduce(self, "ADD", axes);

  self.matmul = (other) => {
    if (self.shape.length !== 2 || other.shape.length !== 2) throw new Error("matmul expects 2D tensors");
    const [m, k] = self.shape;
    const [k2, n] = other.shape;
    if (k !== k2) throw new Error(`matmul shape mismatch: [${m},${k}] x [${k2},${n}]`);

    return mkTensor(
      uop.reduce(
        uop.mul(
          uop.view(self.uop, [{ dims: [m, k, n], strides: [k, 1, 0] }]),
          uop.view(other.uop, [{ dims: [m, k, n], strides: [0, n, 1] }])
        ),
        [1],
        "ADD"
      ),
      [m, n]
    );
  };

  self.reshape = (dims) => {
    if (numel(dims) !== self.numel()) throw new Error("reshape numel mismatch");
    return shapeOp(self, "RESHAPE", dims.slice(), { shape: dims.slice() });
  };

  self.permute = (axes) => {
    if (axes.length !== self.shape.length) throw new Error("permute axes rank mismatch");
    const out = axes.map((a) => self.shape[a]);
    return shapeOp(self, "PERMUTE", out, { shape: axes.slice() });
  };

  self.expand = (dims) => shapeOp(self, "EXPAND", dims.slice(), { shape: dims.slice() });

  self.pad = (pads) =>
    shapeOp(
      self,
      "PAD",
      self.shape.map((d, i) => pads[i][0] + d + pads[i][1]),
      { args: pads }
    );

  self.shrink = (cuts) => shapeOp(self, "SHRINK", cuts.map(([a, b]) => b - a), { args: cuts });

  self.run = async (_backend?: RuntimeName) => {
    const backend = _backend ?? BACKEND.default;
    if (backend !== "webgpu") throw new Error(`backend ${backend} not implemented`);
    const kg = kernelize(self.uop);
    const lg = lowerer(kg);
    if (DEBUG.get()){
      console.log(uop.fmt(kg))
      console.log(uop.fmt(lg))
    }
    if (lg.op !== "KERNEL") throw new Error("expected KERNEL root after kernelize/lowerer");
    const sched = linearize(lg);
    if (sched.length === 0) throw new Error("linearize returned no kernels");

    if (DEBUG.get()) sched.forEach(s=>console.log(s.toString()))


    let out: ReturnType<typeof bufferRefsIn>[number] | null = null;
    const refs = sched.flatMap((s) => bufferRefsIn(s.steps));
    const bindings: Parameters<ReturnType<typeof WEBGPU.createRunner>["run"]>[0] = [];
    for (const r of refs) if (!bindings[r.slot]) bindings[r.slot] = WEBGPU.createBuffer(r.size);

    const kernels = sched.map((s) => WEBGPU.createRunner(s.steps))

    let st = performance.now()
    for (const k of kernels) await k.run(bindings)
    if (DEBUG.get()) console.log(`execution duration: ${(performance.now()- st)/1e3}s`)
    out = outputBufferIn(sched[sched.length-1].steps);
    if (!out) throw new Error("no output buffer found");
    return bindings[out.slot]!.read();
  };
  return self;
};

export const Tensor = {
  const: (value: number, dims: number[]): Tensor =>
    mkTensor(uop.view({ op: "CONST", srcs: [], val: [value] }, [{ dims, strides: stridesFor(dims) }]), dims.slice()),

  new: (raw: Raw = 0): Tensor => {
    const dims = shapeFromRaw(raw);
    return mkTensor(
      uop.view({ op: "CONST", srcs: [], val: flattenRaw(raw) }, [{ dims, strides: stridesFor(dims) }]),
      dims
    );
  },

  rand: (dims: number[]): Tensor =>
    mkTensor({ op: "RAND", srcs: [], seed: (Math.random() * 0x7fffffff) | 0, size: numel(dims) }, dims.slice()),
};
