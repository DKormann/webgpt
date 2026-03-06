import type { BACKEND, RAWBUFFER, UOp } from "./types";
import { DEBUG } from "./debug";
import { uop } from "./uops";

export type WEBGPUBUFFER = RAWBUFFER;
type State = { size: number; gpu?: GPUBuffer };

type Compiled = {
  wgsl: string;
  slots: number[];
  randCount: number;
  dispatch: [number, number, number];
  pipeline?: Promise<GPUComputePipeline>;
};

const states = new WeakMap<WEBGPUBUFFER, State>();
const cache = new Map<string, Compiled>();
let gpuP: Promise<GPU> | undefined;
let devP: Promise<GPUDevice> | undefined;

const getGPU = () =>
  (gpuP ??= (async () => {
    let g = (globalThis as { navigator?: Navigator }).navigator?.gpu;
    if (g) return g;
    const m = await (new Function("m", "return import(m)") as (m: string) => Promise<{ setupGlobals: () => void }>)("bun-webgpu");
    m.setupGlobals();
    g = (globalThis as { navigator?: Navigator }).navigator?.gpu;
    if (!g) throw new Error("bun-webgpu did not expose navigator.gpu");
    return g;
  })());

const getDevice = () =>
  (devP ??= (async () => {
    const a = await (await getGPU()).requestAdapter();
    if (!a) throw new Error("No GPU adapter");
    return a.requestDevice();
  })());

const ensure = async (b: WEBGPUBUFFER) => {
  const st = states.get(b);
  if (!st) throw new Error("Unknown WEBGPU buffer");
  if (st.gpu) return st.gpu;
  const d = await getDevice();
  st.gpu = d.createBuffer({
    size: Math.max(4, st.size * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const init = (b as WEBGPUBUFFER & { __initData?: number[] }).__initData;
  if (init?.length) d.queue.writeBuffer(st.gpu, 0, new Float32Array(init));
  return st.gpu;
};

const read = async (b: WEBGPUBUFFER) => {
  const d = await getDevice();
  const src = await ensure(b);
  const st = states.get(b)!;
  const bytes = Math.max(4, st.size * 4);
  const dst = d.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const ce = d.createCommandEncoder();
  ce.copyBufferToBuffer(src, 0, dst, 0, bytes);
  d.queue.submit([ce.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  const out = Array.from(new Float32Array(dst.getMappedRange().slice(0))).slice(0, st.size);
  dst.unmap();
  dst.destroy();
  return out;
};

const createBuffer = (size: number): WEBGPUBUFFER => {
  const b: WEBGPUBUFFER = { size, read: () => read(b) };
  states.set(b, { size });
  return b;
};

const bodyOf = (g: UOp[]) => (g.length === 1 && g[0].op === "KERNEL" ? g[0].srcs : g);
const randNodesOf = (graph: UOp[]): (UOp & { op: "RAND" })[] => {
  const out: (UOp & { op: "RAND" })[] = [];
  const seen = new Set<UOp>();
  const walk = (u: UOp) => {
    if (seen.has(u)) return;
    seen.add(u);
    if (u.op === "RAND") out.push(u);
    u.srcs.forEach(walk);
  };
  graph.forEach(walk);
  return out;
};
const bufferNodesOf = (graph: UOp[]): (UOp & { op: "BUFFER" })[] => {
  const out: (UOp & { op: "BUFFER" })[] = [];
  const seen = new Set<UOp>();
  const walk = (u: UOp) => {
    if (seen.has(u)) return;
    seen.add(u);
    if (u.op === "BUFFER") out.push(u);
    u.srcs.forEach(walk);
  };
  graph.forEach(walk);
  return out;
};

const codegen = (graphIn: UOp[]): Omit<Compiled, "pipeline"> => {
  const graph = bodyOf(graphIn);
  const slots = Array.from(new Set(bufferNodesOf(graph).map((u) => u.slot))).sort((a, b) => a - b);
  const bindBySlot = new Map(slots.map((s, i) => [s, i]));
  const rands = randNodesOf(graph);
  const randIx = new Map(rands.map((r, i) => [r, i]));
  const seedBinding = slots.length;
  const names = new Map<UOp, string>();
  const regs = new Map<UOp, string>();
  const ranges: UOp[] = [];
  const lines: string[] = [];
  const specials = graph.filter((u): u is UOp & { op: "SPECIAL" } => u.op === "SPECIAL");
  const gid = ["gid.x", "gid.y", "gid.z"] as const;
  if (specials.length > 3) throw new Error("supports at most 3 SPECIAL dims");

  const c = (u: UOp & { op: "CONST" }) => u.val[0] ?? 0;
  const op2 = (u: UOp) => u.op === "ADD" || u.op === "MUL";

  const idx = (u: UOp): string => {
    if (u.op === "CONST") return `${c(u)}u`;
    if (u.op === "RANGE" || u.op === "SPECIAL") return names.get(u) ?? (() => { throw new Error("range used outside scope"); })();
    if (op2(u)) {
      const [a, b] = u.srcs as [UOp, UOp];
      return `(${idx(a)} ${u.op === "ADD" ? "+" : "*"} ${idx(b)})`;
    }
    throw new Error(`unsupported index arg: ${u.op}`);
  };

  const val = (u: UOp): string => {
    if (u.op === "CONST") {
      const v = c(u);
      return Number.isInteger(v) ? `${v}.0` : String(v);
    }
    if (u.op === "RANGE" || u.op === "SPECIAL") return `f32(${names.get(u) ?? (() => { throw new Error("range used outside scope"); })()})`;
    if (op2(u)) {
      const [a, b] = u.srcs as [UOp, UOp];
      return `(${val(a)} ${u.op === "ADD" ? "+" : "*"} ${val(b)})`;
    }
    if (u.op === "DEFINE_REG") return regs.get(u) ?? (() => { throw new Error("register used before declaration"); })();
    if (u.op === "NOOP") {
      const ch = u.srcs[0];
      if (op2(ch) && ch.srcs[0]?.op === "DEFINE_REG") return regs.get(ch.srcs[0]) ?? (() => { throw new Error("register read with undeclared register"); })();
      return val(ch);
    }
    if (u.op === "INDEX") {
      const [base, i] = u.srcs;
      if (base.op === "RAND") {
        const ri = randIx.get(base);
        if (ri == null) throw new Error("missing RAND seed binding");
        return `randf(seeds[${ri}u] ^ ${idx(i)})`;
      }
      if (base.op !== "BUFFER") throw new Error(`unsupported value index base: ${base.op}`);
      const b = bindBySlot.get(base.slot);
      if (b == null) throw new Error("graph references unknown buffer slot");
      return `b${b}[${idx(i)}]`;
    }
    throw new Error(`unsupported store src: ${u.op}`);
  };

  let s = 0;
  for (const u of graph) {
    if (u.op === "SPECIAL") {
      names.set(u, gid[s++]);
      continue;
    }
    if (u.op === "RANGE") {
      const n = `r${ranges.length}`;
      names.set(u, n);
      ranges.push(u);
      lines.push(`for (var ${n}:u32 = 0u; ${n} < ${u.max}u; ${n} = ${n} + 1u) {`);
      continue;
    }
    if (u.op === "ENDRANGE") {
      if (ranges.pop() !== u.srcs[0]) throw new Error("ENDRANGE mismatch");
      lines.push("}");
      continue;
    }
    if (u.op === "DEFINE_REG") {
      const r = `reg${regs.size}`;
      regs.set(u, r);
      lines.push(`var ${r}: f32 = ${u.default};`);
      continue;
    }
    if (op2(u)) {
      const [a, b] = u.srcs as [UOp, UOp];
      if (a.op !== "DEFINE_REG") continue;
      const r = regs.get(a);
      if (!r) throw new Error("register update with undeclared register");
      lines.push(`${r} = ${u.op === "ADD" ? `${r} + ${val(b)}` : `${r} * ${val(b)}`};`);
      continue;
    }
    if (u.op === "NOOP" || u.op === "KERNEL" || u.op === "RAND" || u.op === "BUFFER" || u.op === "CONST" || u.op === "INDEX" || op2(u)) continue;
    if (u.op !== "STORE") throw new Error(`unsupported root op: ${u.op}`);
    const [src, dst] = u.srcs;
    if (dst.op !== "INDEX" || dst.srcs[0].op !== "BUFFER") throw new Error("unsupported store dst");
    const b = bindBySlot.get(dst.srcs[0].slot);
    if (b == null) throw new Error("graph references unknown buffer slot");
    lines.push(`b${b}[${idx(dst.srcs[1])}] = ${val(src)};`);
  }
  while (ranges.length) {
    ranges.pop();
    lines.push("}");
  }

  const needRand = rands.length > 0;
  const guard = specials.map((u) => `${gid[u.axis]} >= ${u.extent}u`).join(" || ");
  const threadExt = [1, 1, 1] as [number, number, number];
  for (const s0 of specials) threadExt[s0.axis] = s0.thread;
  const dispatch: [number, number, number] = [1, 1, 1];
  for (const s0 of specials) dispatch[s0.axis] = Math.max(1, s0.block);

  return {
    slots,
    randCount: rands.length,
    dispatch,
    wgsl: [
      ...slots.map((_, i) => `@group(0) @binding(${i}) var<storage, read_write> b${i}: array<f32>;`),
      ...(needRand ? [`@group(0) @binding(${seedBinding}) var<storage, read> seeds: array<u32>;`] : []),
      ...(needRand
        ? [
            "fn randf(x:u32) -> f32 {",
            "  var z = x + 0x9e3779b9u;",
            "  z = (z ^ (z >> 16u)) * 0x85ebca6bu;",
            "  z = (z ^ (z >> 13u)) * 0xc2b2ae35u;",
            "  z = z ^ (z >> 16u);",
            "  return f32(z) * 2.3283064365386963e-10;",
            "}",
          ]
        : []),
      `@compute @workgroup_size(${threadExt[0]}, ${threadExt[1]}, ${threadExt[2]})`,
      "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
      ...(guard ? [`  if (${guard}) { return; }`] : []),
      ...lines.map((l) => `  ${l}`),
      "}",
    ].join("\n"),
  };
};

export const WEBGPU: BACKEND<WEBGPUBUFFER> = {
  max_blocks: [65535, 65535, 65535],
  max_threads: [256, 256, 64],
  createBuffer,
  createRunner: (graph: UOp[]) => {
    const key = JSON.stringify(bodyOf(graph).map((g) => uop.hash(g)));
    if (!cache.has(key)) cache.set(key, codegen(graph));
    const compiled = cache.get(key)!;
    if (DEBUG.get() === 1) console.log(compiled.wgsl);

    const run = async (bindings: RAWBUFFER[]) => {
      const d = await getDevice();
      const gbufs = await Promise.all(
        compiled.slots.map((slot) => {
          const b = bindings[slot] as WEBGPUBUFFER | undefined;
          if (!b) throw new Error(`missing binding for buffer slot ${slot}`);
          return ensure(b);
        })
      );

      const seedBuf =
        compiled.randCount > 0
          ? d.createBuffer({ size: Math.max(4, compiled.randCount * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
          : null;
      if (seedBuf) {
        const rands = randNodesOf(bodyOf(graph));
        d.queue.writeBuffer(seedBuf, 0, new Uint32Array(rands.map((r) => Math.floor(r.seed) >>> 0)));
      }

      if (!compiled.pipeline) {
        compiled.pipeline = (async () => {
          const m = d.createShaderModule({ code: compiled.wgsl });
          return d.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: "main" } });
        })();
      }
      const p = await compiled.pipeline;
      const bg = d.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: [
          ...gbufs.map((buffer, i) => ({ binding: i, resource: { buffer } })),
          ...(seedBuf ? [{ binding: compiled.slots.length, resource: { buffer: seedBuf } }] : []),
        ],
      });

      const ce = d.createCommandEncoder();
      const pass = ce.beginComputePass();
      pass.setPipeline(p);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(compiled.dispatch[0], compiled.dispatch[1], compiled.dispatch[2]);
      pass.end();
      d.queue.submit([ce.finish()]);
      seedBuf?.destroy();
    };

    return { graph, run };
  },
};
