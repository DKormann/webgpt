import type { Backend, BufferRef, Linear, Programm, RAWBUFFER, UOp } from "./types";
import { DEBUG } from "./debug";

import { uop } from "./uops";

export type WEBGPUBUFFER = RAWBUFFER;
type State = { size: number; gpu?: GPUBuffer };

type Compiled = {
  wgsl: string;
  buffers: BufferRef[],
  randCount: number;
  dispatch: [number, number, number];
  pipeline: Promise<GPUComputePipeline>;
};

const states = new WeakMap<WEBGPUBUFFER, State>();
let gpuP: Promise<GPU> | undefined;
let devP: Promise<GPUDevice> | undefined;

const getGPU = () : Promise<GPU> =>
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
  // const slots = Array.from(new Set(bufferNodesOf(graph).map((u) => u.slot))).sort((a, b) => a - b);
  let buffers:BufferRef[] = Array.from(new Set(bufferNodesOf(graph)))
  const bindBySlot = new Map(buffers.map((s, i) => [s, i]));
  const written = new Set<BufferRef>();

  const rands = randNodesOf(graph);
  const randIx = new Map(rands.map((r, i) => [r, i]));
  const seedBinding = buffers.length;
  const names = new Map<UOp, string>();
  const regs = new Map<UOp, string>();
  const ranges: UOp[] = [];
  const lines: string[] = [];
  const specials = graph.filter((u): u is UOp & { op: "SPECIAL" } => u.op === "SPECIAL");
  const gid = ["gid.x", "gid.y", "gid.z"] as const;
  if (specials.length > 3) throw new Error("supports at most 3 SPECIAL dims");

  // lines.push("hello")

  let addreg = (c:string, u:UOp) =>{
    let name = 'x' + names.size;
    lines.push(`${u.op == "DEFINE_REG" ? "var" : "let"} ${name}: ${gettype(u)} = ${c};`)
    names.set(u, name)
  }

  let addbin = (op:string, u:UOp) => addreg(`${names.get(u.srcs[0]!)} ${op} ${names.get(u.srcs[1]!)}`, u)

  let gettype = (u:UOp) =>{
    if (u.op == "RANGE") return "u32";
    if (u.op == "INDEX") return "f32";
    if (u.op == "RAND") return "f32";
    if (u.op == "SPECIAL") return "u32";
    if (u.op == "DEFINE_REG") return "f32";
    if (u.op == "ADD" || u.op == "MUL") return gettype(u.srcs[0]) 
    return "UNK DTYPE"
  }

  graph.forEach(u=>{
    if (u.op == "DEFINE_REG") addreg("0", u)
    else if (u.op == "ADD") addbin("+", u)
    else if (u.op == "MUL") addbin("*", u)
    else if (u.op == "INDEX") names.set(u, `${names.get(u.srcs[0])}[${names.get(u.srcs[1])}]`)
    else if (u.op == "RAND") {
      const ri = randIx.get(u);
      if (ri == null) throw new Error("missing RAND seed binding");
      addreg(`randf(seeds[${ri}u] ^ ${names.get(u.srcs[0]!) ?? "0u"})`, u)
    }
    else if (u.op == "BUFFER") names.set(u,`b${buffers.indexOf(u)}`)
    else if (u.op == "CONST") names.set(u, String(u.val[0]))
    else if (u.op == "RANGE") {
      lines.push(`for (var r = 0u; r < ${u.max}; r ++){`)
      addreg('r', u)
    }
    else if (u.op == "STORE"){

      uop.topo(u.srcs[1]).filter(x=>x.op == "BUFFER").forEach(b=>written.add(b))
      lines.push(`${names.get(u.srcs[1])} = ${names.get(u.srcs[0])};`)
    }else if (u.op == "NOOP") names.set(u, names.get(u.srcs[0])!)
    else if (u.op == "ENDRANGE") lines.push("}")

    else if (u.op == "SPECIAL") names.set(u, `${gid[u.axis]}`)
    else if ( u.op == "KERNEL") return
    else return lines.push(u.op)

  })




  const needRand = rands.length > 0;
  const guard = specials.map((u) => `${gid[u.axis]} >= ${u.extent}u`).join(" || ");
  const threadExt = [1, 1, 1] as [number, number, number];
  for (const s0 of specials) threadExt[s0.axis] = s0.thread;
  const dispatch: [number, number, number] = [1, 1, 1];
  for (const s0 of specials) dispatch[s0.axis] = Math.max(1, s0.block);

  const matmulFast = (() => {
    if (rands.length) return null;
    const sx = specials.find((s) => s.axis === 0);
    const sy = specials.find((s) => s.axis === 1);
    const red = graph.find((u): u is UOp & { op: "RANGE" } => u.op === "RANGE");
    const accStore = graph.find((u): u is UOp & { op: "STORE" } => u.op === "STORE" && u.srcs[1]?.op === "DEFINE_REG");
    const outStore = [...graph].reverse().find((u): u is UOp & { op: "STORE" } => u.op === "STORE" && u.srcs[1]?.op === "INDEX" && u.srcs[1].srcs[0]?.op === "BUFFER");
    if (!sx || !sy || !red || !accStore || !outStore) return null;
    const add = accStore.srcs[0];
    if (add.op !== "ADD") return null;
    const mul = add.srcs.find((s): s is UOp & { op: "MUL" } => s.op === "MUL");
    if (!mul) return null;
    const [ia, ib] = mul.srcs;
    if (ia.op !== "INDEX" || ib.op !== "INDEX") return null;
    const ba = ia.srcs[0], bb = ib.srcs[0];
    if (ba.op !== "BUFFER" || bb.op !== "BUFFER") return null;
    const outBase = outStore.srcs[1].srcs[0];
    if (outBase.op !== "BUFFER") return null;
    const mats = Array.from(new Set([ba, bb, outBase])) as BufferRef[];
    const bind = new Map(mats.map((b, i) => [b, i]));
    const M = sx.extent, N = sy.extent, K = red.max;
    const T = 16;
    const tiles = Math.ceil(K / T);
    const wgsl = [
      ...mats.map((b, i) => `@group(0) @binding(${i}) var<storage, ${b === outBase ? "read_write" : "read"}> b${i}: array<f32>;`),
      `var<workgroup> Asub: array<f32, ${T * T}>;`,
      `var<workgroup> Bsub: array<f32, ${T * T}>;`,
      `@compute @workgroup_size(${T}, ${T}, 1)`,
      "fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {",
      "  let row: u32 = gid.x;",
      "  let col: u32 = gid.y;",
      "  var acc: f32 = 0.0;",
      `  for (var t:u32 = 0u; t < ${tiles}u; t = t + 1u) {`,
      `    let kA: u32 = t * ${T}u + lid.y;`,
      `    let kB: u32 = t * ${T}u + lid.x;`,
      `    let li: u32 = lid.x * ${T}u + lid.y;`,
      `    if (row < ${M}u && kA < ${K}u) { Asub[li] = b${bind.get(ba)!}[row * ${K}u + kA]; } else { Asub[li] = 0.0; }`,
      `    if (kB < ${K}u && col < ${N}u) { Bsub[li] = b${bind.get(bb)!}[kB * ${N}u + col]; } else { Bsub[li] = 0.0; }`,
      "    workgroupBarrier();",
      `    for (var kk:u32 = 0u; kk < ${T}u; kk = kk + 1u) {`,
      `      acc = acc + Asub[lid.x * ${T}u + kk] * Bsub[kk * ${T}u + lid.y];`,
      "    }",
      "    workgroupBarrier();",
      "  }",
      `  if (row < ${M}u && col < ${N}u) { b${bind.get(outBase)!}[row * ${N}u + col] = acc; }`,
      "}",
    ].join("\n");
    return {
      buffers: mats,
      randCount: 0,
      dispatch: [Math.ceil(M / T), Math.ceil(N / T), 1] as [number, number, number],
      wgsl,
    };
  })();
  if (matmulFast) return matmulFast;

  return {
    buffers,
    randCount: rands.length,
    dispatch,
    wgsl: [

      ...buffers.map((b,i)=>`@group(0) @binding(${i}) var<storage, ${written.has(b) ? "read_write" : "read"}> b${i}: array<f32>;`),
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

const mkKernel = (d:GPUDevice, {srcs:graph}:Linear) =>{

    const compiled = codegen(graph)
    if (DEBUG.get()) console.log(compiled.wgsl)

    const seedBuf =
      compiled.randCount > 0
        ? d.createBuffer({ size: Math.max(4, compiled.randCount * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
        : null;
    if (seedBuf) {
      const rands = randNodesOf(bodyOf(graph));
      d.queue.writeBuffer(seedBuf, 0, new Uint32Array(rands.map((r) => Math.floor(r.arg.seed) >>> 0)));
    }

    const pipeline = d.createComputePipeline({ layout: "auto", compute: { module: d.createShaderModule({ code: compiled.wgsl }), entryPoint: "main" } });
    const run = async (getBind: (b:BufferRef) =>GPUBuffer) => {

      const bg = d.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          ...compiled.buffers.map((buffer, i) => ({ binding: i, resource: { buffer: getBind(buffer) } })),
          ...(seedBuf ? [{ binding: compiled.buffers.length, resource: { buffer: seedBuf } }] : []),
        ],
      });

      const ce = d.createCommandEncoder();
      const pass = ce.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(compiled.dispatch[0], compiled.dispatch[1], compiled.dispatch[2]);
      pass.end();
      d.queue.submit([ce.finish()]);
    }

    return run

}

export const WEBGPU: Backend<WEBGPUBUFFER> = {
  max_blocks: [65535, 65535, 65535],
  max_threads: [256, 256, 64],
  createBuffer,
  createRunner: async (graph: Programm) => {

    const buffers = new Map<number, BufferRef>();
    uop.topo(graph)
      .filter((b): b is BufferRef => b.op == "BUFFER")
      .forEach((b) => { if (!buffers.has(b.arg.slot)) buffers.set(b.arg.slot, b); });
    const d = await getDevice();
    let kerns = graph.srcs.map(x=>mkKernel(d,x))

    return async (getBind : (ref: BufferRef) => WEBGPUBUFFER) => {
      const gpubuffers = new Map<number, GPUBuffer>(
        await Promise.all(
          Array.from(buffers.entries()).map(async ([slot, br]) => [slot, await ensure(getBind(br))] as [number, GPUBuffer])
        )
      );
      for (let k of kerns) await k((r) => {
        const g = gpubuffers.get(r.arg.slot);
        if (!g) throw new Error(`GPU buffer slot ${r.arg.slot} not bound`);
        return g;
      })
      await d.queue.onSubmittedWorkDone();
    };
  },
};
