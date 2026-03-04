import type { UOp, TensorShape, BinOp, Schedule, Shape } from "./types";
import { uop } from "./uops";
import { kernelize } from "./kernelize";
// import { lower } from "./lower";
import { linearize } from "./linearize";
import { WEBGPU } from "./webgpu";
import { DEBUG } from "./debug";
import { log } from "./helpers";

export type Raw = number | Raw[];
export type RuntimeName = "js" | "webgpu";

export type Tensor = {
  uop: UOp;
  shape: number[];
  numel: ()=>number;
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

export const BACKEND: { default: RuntimeName } = {default:"webgpu"};

const mkContiguousShape = (dims: number[]): TensorShape => ({
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
  return mkContiguousShape(dims);
};

const binary = (self:Tensor, op: BinOp) => (other:Tensor) => {
  if (JSON.stringify(self.shape) != JSON.stringify(other.shape)) throw new Error("shape mismatch");
  return mkTensor({ op, srcs: [self.uop, other.uop] }, self.shape);
};

const normalizeAxes = (axes: number[] | undefined, rank: number): number[] => {
  const raw = axes ?? [...Array(rank).keys()];
  const norm = raw.map((a) => (a < 0 ? rank + a : a));
  for (const a of norm) {
    if (a < 0 || a >= rank) throw new Error(`sum axis out of range: ${a} for rank ${rank}`);
  }
  return [...new Set(norm)].sort((a, b) => b - a);
};

const buffersIn = (graph: UOp[]): Set<UOp & { op: "BUFFER" }> => {
  const out = new Set<UOp & { op: "BUFFER" }>();
  const walk = (u: UOp) => {
    if (u.op === "BUFFER") out.add(u);
    u.srcs.forEach(walk);
  };
  graph.forEach(walk);
  return out;
};



let reduce = (self:Tensor, op:BinOp, axes?: number[])=>{

  if (axes==undefined) axes = self.shape.map((_, i) => i)

  return mkTensor(
    uop.reduce(
      self.uop,
      axes,
      op
    ),
    self.shape.filter((x,i)=>!axes.includes(i))
  )
}


const mkTensor = (graph: UOp, shape: number[]): Tensor => {
  const self = {} as Tensor;
  self.uop = graph;
  self.shape = shape;


  self.add = binary(self, "ADD")
  self.mul = binary(self, "MUL")
  self.sum = (axes?) => reduce(self, "ADD", axes)
  // self.prod = (axes?) => reduce(self, "MUL", axes)


  self.matmul = (other) => {
    if (self.shape.dims.length !== 2 || other.shape.dims.length !== 2) {
      throw new Error("matmul expects 2D tensors");
    }
    const [m, k] = self.shape.dims;
    const [k2, n] = other.shape.dims;
    if (k !== k2) throw new Error(`matmul shape mismatch: [${m},${k}] x [${k2},${n}]`);
  
    return mkTensor(
      uop.reduce(
        uop.mul(
          uop.view(self.uop, [{ dims: [m, k, n], strides: [k, 1, 0] }]),
          uop.view(other.uop, [{ dims: [m, k, n], strides: [0, n, 1] }])
        ),
        1, "ADD"
      ),
      mkContiguousShape([m, n])
    );
  };

  self.reshape = (dims) => mkTensor(self.uop, mkContiguousShape(dims));

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


    const logSchedule = (x:Schedule, name ="")=> x.items.forEach(x=>{
      console.log(` ======= SCHEDULE ITEM: ${name} ======= `)
      x.roots.forEach(u=>console.log(uop.fmt(u)))
    })

    const backend = _backend ?? BACKEND.default;
    if (backend !== "webgpu") throw new Error(`backend ${backend} not implemented`);
    const sched = kernelize(self, WEBGPU.createBuffer);

    if(DEBUG.get()) logSchedule(sched)

    // const lowSched = lower(sched, (size, data) => {
    //   const b = WEBGPU.createBuffer(size) as typeof WEBGPU extends { createBuffer: (...args: any[]) => infer T } ? T : never;
    //   if (data) (b as { __initData?: number[] }).__initData = data.slice();
    //   return b;
    // });

    // if(DEBUG.get()) logSchedule(lowSched, "lower")


    let out: number[] = [];
    for (const item of sched.items) {
      for (const root of item.roots) {
        const low = linearize(root);
        const used = buffersIn(low);
        const bufs = item.Buffers.filter((b) => Array.from(used).some((u) => u.buf === b));
        const k = WEBGPU.createKernel(low, bufs as Parameters<typeof WEBGPU.createKernel>[1]);
        await k.launch();
      }
      out = await item.Buffers[0].read();
    }
    return out;
  };

  return self;
};

export const Tensor = {
  const: (value: number, dims: number[]): Tensor => {
    let shape = {
      dims,
      strides: dims.map(x=>0)
    }  as TensorShape;
    return mkTensor(
      uop.view({ op: "CONST", srcs: [], val: [value] }, [shape]),
      shape
    );
  },

  new: (raw: Raw = 0): Tensor => {
    const shape = shapeFromRaw(raw);
    return mkTensor(
      uop.view({ op: "CONST", srcs: [], val: flattenRaw(raw) }, [{ dims: shape.dims, strides: shape.strides }]),
      shape
    );
  },

  rand: (dims: number[]): Tensor => {
    const shape = mkContiguousShape(dims);
    return mkTensor({ op: "RAND", srcs: [], seed: (Math.random() * 0x7fffffff) | 0, size: shape.numel }, shape);
  }
};
