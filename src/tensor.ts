
import { execAsync, type RuntimeName } from "./runtime/index.ts";
import { graphUOP, type UOPGraphItem, type Shape, type UOP } from "./uops.ts";
import { linear as nnLinear } from "./nn.ts";

export type Raw = number | Raw[];
export type TensorOpts = { requiresGrad?: boolean };

type TensorCtx = {
  backward: (gradOut: Tensor) => void;
};


type TensorMethods = {
  add: (b: Tensor) => Tensor;
  mul: (b: Tensor) => Tensor;
  sum: (dims?: number[]) => Tensor;
  prod: (dims?: number[]) => Tensor;
  reshape: (dims: number[]) => Tensor;
  permute: (axes: number[]) => Tensor;
  expand: (dims: number[]) => Tensor;
  pad: (pads: [number, number][]) => Tensor;
  shrink: (cuts: [number, number][]) => Tensor;
  run: (backend?: RuntimeName) => Promise<Raw>;
  backward: (seed?: Tensor) => void;
  zeroGrad: () => void;
  detach: () => Tensor;
  matmul: (b: Tensor) => Tensor;
  graph: () => UOPGraphItem[];
};

export type TensorData = {
  uop: UOP;
  shape: Shape;
  requiresGrad: boolean;
  grad: Tensor | null;
  _ctx: TensorCtx | null;
  _parents: Tensor[];
};

export type Tensor = TensorData & TensorMethods;

export const BACKEND: { default?: RuntimeName } = { default:undefined };

const mkShape = (dims: number[]): Shape => ({
  dims,
  strides: dims.map((_, i) => dims.slice(i + 1).reduce((a, c) => a * c, 1)),
  numel: dims.reduce((a, c) => a * c, 1)
});

const nest = (flat: number[], dims: number[]): Raw => {
  if (dims.length === 0) return flat[0] ?? 0;
  if (dims.length === 1) return flat.slice(0, dims[0]);
  const step = dims.slice(1).reduce((a, c) => a * c, 1);
  const out: Raw[] = [];
  for (let i = 0; i < dims[0]; i++) out.push(nest(flat.slice(i * step, (i + 1) * step), dims.slice(1)));
  return out;
};

const reduceDims = (shape: Shape, dims?: number[]): number[] => {
  const n = shape.dims.length;
  const base = dims && dims.length ? dims : [...Array(n)].map((_, i) => i);
  return [...new Set(base.map((d) => (d < 0 ? n + d : d)))].sort((a, b) => a - b);
};

const reduceShape = (shape: Shape, dims: number[]): Shape => {
  const drop = new Set(dims);
  return mkShape(shape.dims.filter((_, i) => !drop.has(i)));
};

const restoreReducedDims = (outGrad: Tensor, inShape: Shape, reduced: number[]): Tensor => {
  const reducedSet = new Set(reduced);
  const reshaped: number[] = [];
  let j = 0;
  for (let i = 0; i < inShape.dims.length; i++) {
    if (reducedSet.has(i)) reshaped.push(1);
    else reshaped.push(outGrad.shape.dims[j++] ?? 1);
  }
  return outGrad.reshape(reshaped).expand(inShape.dims);
};

const addGrad = (target: Tensor, grad: Tensor) => {
  const g = grad.detach();
  target.grad = target.grad ? target.grad.add(g).detach() : g;
};

const flattenRaw = (raw: Raw): number[] => ([raw] as number[]).flat(Infinity) as number[];

const shapeFromRaw = (raw: Raw): Shape => {
  const dims: number[] = [];
  let cur: Raw = raw;
  while (Array.isArray(cur)) {
    dims.push(cur.length);
    cur = cur[0] as Raw;
  }
  return mkShape(dims);
};

const mkTensor = (init: { uop: UOP; shape: Shape; requiresGrad?: boolean }): Tensor => {
  const self = {} as Tensor;
  self.uop = init.uop;
  self.shape = init.shape;
  self.requiresGrad = !!init.requiresGrad;
  self.grad = null;
  self._ctx = null;
  self._parents = [];

  self.detach = () => mkTensor({ uop: self.uop, shape: self.shape, requiresGrad: false });

  self.add = (b) => {
    const out = mkTensor({
      uop: { op: "ADD", srcs: [self.uop, b.uop], srcShapes: [self.shape, b.shape] },
      shape: self.shape,
      requiresGrad: self.requiresGrad || b.requiresGrad
    });
    if (out.requiresGrad) {
      out._parents = [self, b];
      out._ctx = {
        backward: (go) => {
          if (self.requiresGrad) addGrad(self, go);
          if (b.requiresGrad) addGrad(b, go);
        }
      };
    }
    return out;
  };

  self.mul = (b) => {
    const out = mkTensor({
      uop: { op: "MUL", srcs: [self.uop, b.uop], srcShapes: [self.shape, b.shape] },
      shape: self.shape,
      requiresGrad: self.requiresGrad || b.requiresGrad
    });
    if (out.requiresGrad) {
      out._parents = [self, b];
      out._ctx = {
        backward: (go) => {
          if (self.requiresGrad) addGrad(self, go.mul(b.detach()));
          if (b.requiresGrad) addGrad(b, go.mul(self.detach()));
        }
      };
    }
    return out;
  };

  self.sum = (dims) => {
    const rd = reduceDims(self.shape, dims);
    const out = mkTensor({
      uop: { op: "REDUCE", bin: "ADD", src: self.uop, inShape: self.shape, dims: rd },
      shape: reduceShape(self.shape, rd),
      requiresGrad: self.requiresGrad
    });
    if (out.requiresGrad) {
      out._parents = [self];
      out._ctx = {
        backward: (go) => addGrad(self, restoreReducedDims(go, self.shape, rd))
      };
    }
    return out;
  };

  self.prod = (dims) => {
    const rd = reduceDims(self.shape, dims);
    const out = mkTensor({
      uop: { op: "REDUCE", bin: "MUL", src: self.uop, inShape: self.shape, dims: rd },
      shape: reduceShape(self.shape, rd),
      requiresGrad: self.requiresGrad
    });
    if (out.requiresGrad) {
      out._parents = [self];
      out._ctx = {
        backward: () => {
          throw new Error("prod backward not implemented yet");
        }
      };
    }
    return out;
  };

  self.reshape = (dims) => {
    const out = mkTensor({ uop: self.uop, shape: mkShape(dims), requiresGrad: self.requiresGrad });
    if (out.requiresGrad) {
      out._parents = [self];
      out._ctx = { backward: (go) => addGrad(self, go.reshape(self.shape.dims)) };
    }
    return out;
  };

  self.permute = (axes) => {
    const out = mkTensor({
      uop: self.uop,
      shape: {
        dims: axes.map((a) => self.shape.dims[a]),
        strides: axes.map((a) => self.shape.strides[a]),
        numel: self.shape.numel,
        offset: self.shape.offset,
        mask: self.shape.mask ? axes.map((a) => self.shape.mask![a]) : undefined
      },
      requiresGrad: self.requiresGrad
    });
    if (out.requiresGrad) {
      const inv = new Array(axes.length).fill(0);
      for (let i = 0; i < axes.length; i++) inv[axes[i]] = i;
      out._parents = [self];
      out._ctx = { backward: (go) => addGrad(self, go.permute(inv)) };
    }
    return out;
  };

  self.expand = (dims) => {
    const out = mkTensor({
      uop: self.uop,
      shape: {
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
      },
      requiresGrad: self.requiresGrad
    });
    if (out.requiresGrad) {
      const redDims = dims
        .map((d, i) => ({ d, i }))
        .filter(({ d, i }) => (self.shape.dims[i] ?? 1) === 1 && d > 1)
        .map(({ i }) => i);
      out._parents = [self];
      out._ctx = {
        backward: (go) => {
          const g = redDims.length ? go.sum(redDims).reshape(self.shape.dims) : go;
          addGrad(self, g);
        }
      };
    }
    return out;
  };

  self.pad = (pads) => {
    const out = mkTensor({
      uop: self.uop,
      shape: {
        dims: self.shape.dims.map((d, i) => pads[i][0] + d + pads[i][1]),
        strides: self.shape.strides,
        numel: self.shape.dims.map((d, i) => pads[i][0] + d + pads[i][1]).reduce((a, c) => a * c, 1),
        offset: (self.shape.offset ?? 0) - pads.reduce((a, p, i) => a + p[0] * self.shape.strides[i], 0),
        mask: self.shape.dims.map((d, i) => [pads[i][0], pads[i][0] + d])
      },
      requiresGrad: self.requiresGrad
    });
    if (out.requiresGrad) {
      const cuts: [number, number][] = self.shape.dims.map((d, i) => [pads[i][0], pads[i][0] + d]);
      out._parents = [self];
      out._ctx = { backward: (go) => addGrad(self, go.shrink(cuts)) };
    }
    return out;
  };

  self.shrink = (cuts) => {
    const out = mkTensor({
      uop: self.uop,
      shape: {
        dims: cuts.map((c) => c[1] - c[0]),
        strides: self.shape.strides,
        numel: cuts.map((c) => c[1] - c[0]).reduce((a, c) => a * c, 1),
        offset: (self.shape.offset ?? 0) + cuts.reduce((a, c, i) => a + c[0] * self.shape.strides[i], 0)
      },
      requiresGrad: self.requiresGrad
    });
    if (out.requiresGrad) {
      const pads: [number, number][] = cuts.map((c, i) => [c[0], self.shape.dims[i] - c[1]]);
      out._parents = [self];
      out._ctx = { backward: (go) => addGrad(self, go.pad(pads)) };
    }
    return out;
  };

  self.run = async (backend?: RuntimeName) => {
    const flat = await execAsync(backend ?? BACKEND.default ?? "js", self.uop, self.shape);
    return nest(flat, self.shape.dims);
  };

  self.zeroGrad = () => {
    self.grad = null;
  };

  self.backward = (seed?: Tensor) => {
    if (!self.requiresGrad) throw new Error("backward called on tensor with requiresGrad=false");
    const s = seed ?? Tensor.const(1, self.shape.dims);
    addGrad(self, s);

    const topo: Tensor[] = [];
    const seen = new Set<Tensor>();
    const visit = (t: Tensor) => {
      if (seen.has(t)) return;
      seen.add(t);
      for (const p of t._parents) visit(p);
      topo.push(t);
    };
    visit(self);

    for (let i = topo.length - 1; i >= 0; i--) {
      const t = topo[i];
      if (t._ctx && t.grad) t._ctx.backward(t.grad);
    }
  };

  self.matmul = (other) => {
    if (self.shape.dims.length !== 2 || other.shape.dims.length !== 2) {
      throw new Error("matmul expects 2D tensors");
    }
    const [m, k] = self.shape.dims;
    const [k2, n] = other.shape.dims;
    if (k !== k2) throw new Error(`matmul shape mismatch: [${m},${k}] x [${k2},${n}]`);

    const mnk = m * n * k;
    const out = mkTensor({
      uop: {
        op: "REDUCE",
        bin: "ADD",
        src: {
          op: "MUL",
          srcs: [self.uop, other.uop],
          srcShapes: [
            { dims: [m, k, n], strides: [k, 1, 0], numel: mnk },
            { dims: [m, k, n], strides: [0, n, 1], numel: mnk }
          ]
        },
        inShape: { dims: [m, k, n], strides: [k * n, n, 1], numel: mnk },
        dims: [1]
      },
      shape: mkShape([m, n]),
      requiresGrad: self.requiresGrad || other.requiresGrad
    });

    if (out.requiresGrad) {
      out._parents = [self, other];
      out._ctx = {
        backward: (go) => {
          if (self.requiresGrad) addGrad(self, go.matmul(other.permute([1, 0]).detach()));
          if (other.requiresGrad) addGrad(other, self.permute([1, 0]).detach().matmul(go));
        }
      };
    }
    return out;
  };
  self.graph = () => graphUOP(self.uop);

  return self;
};

export const Tensor = {
  const: (value: number, dims: number[], opts: TensorOpts = {}): Tensor =>
    mkTensor({ uop: { op: "CONST", data: [value] }, shape: mkShape(dims), requiresGrad: !!opts.requiresGrad }),

  rand: (dims: number[], opts: TensorOpts = {}): Tensor => {
    const shape = mkShape(dims);
    return mkTensor({
      uop: { op: "RAND", seed: (Math.random() * 0x7fffffff) | 0 },
      shape,
      requiresGrad: !!opts.requiresGrad
    });
  },

  new: (raw: Raw = 0, opts: TensorOpts = {}): Tensor =>
    mkTensor({
      uop: { op: "CONST", data: flattenRaw(raw) },
      shape: shapeFromRaw(raw),
      requiresGrad: !!opts.requiresGrad
    }),

  linear: (input: Tensor, weight: Tensor, bias?: Tensor): Tensor => nnLinear(input, weight, bias),

  graph: (x: Tensor | UOP): UOPGraphItem[] => graphUOP("uop" in x ? x.uop : x),
  logGraph: (x: Tensor | UOP): void => console.table(graphUOP("uop" in x ? x.uop : x))
};
