import type { BACKEND, Kernel, LowGraph, RAWBUFFER, UOp } from "./types";
import { DEBUG } from "./debug";

export type WEBGPUBUFFER = RAWBUFFER & {};

type GPUState = { size: number; gpu?: GPUBuffer };

const states = new WeakMap<WEBGPUBUFFER, GPUState>();

let gpuPromise: Promise<GPU> | null = null;
let devicePromise: Promise<GPUDevice> | null = null;

const getGPU = async (): Promise<GPU> => {
  if (gpuPromise) return gpuPromise;
  gpuPromise = (async () => {
    const g = (globalThis as { navigator?: Navigator }).navigator?.gpu;
    if (g) return g;

    const dynamicImport = new Function("m", "return import(m)") as (m: string) => Promise<{ setupGlobals: () => void }>;
    const m = await dynamicImport("bun-webgpu");
    m.setupGlobals();
    const gg = (globalThis as { navigator?: Navigator }).navigator?.gpu;
    if (!gg) throw new Error("bun-webgpu did not expose navigator.gpu");
    return gg;
  })();
  return gpuPromise;
};

const getDevice = async (): Promise<GPUDevice> => {
  if (devicePromise) return devicePromise;
  devicePromise = (async () => {
    const gpu = await getGPU();
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter");
    return adapter.requestDevice();
  })();
  return devicePromise;
};

const ensureGPUBuffer = async (buffer: WEBGPUBUFFER): Promise<GPUBuffer> => {
  const st = states.get(buffer);
  if (!st) throw new Error("Unknown WEBGPU buffer");
  if (st.gpu) return st.gpu;

  const device = await getDevice();
  st.gpu = device.createBuffer({
    size: Math.max(4, st.size * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  const initData = (buffer as WEBGPUBUFFER & { __initData?: number[] }).__initData;
  if (initData && initData.length) {
    device.queue.writeBuffer(st.gpu, 0, new Float32Array(initData));
  }
  return st.gpu;
};

const readBuffer = async (buffer: WEBGPUBUFFER): Promise<number[]> => {
  const device = await getDevice();
  const gpu = await ensureGPUBuffer(buffer);
  const st = states.get(buffer)!;
  const bytes = Math.max(4, st.size * 4);
  const readback = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(gpu, 0, readback, 0, bytes);
  device.queue.submit([enc.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return Array.from(out).slice(0, st.size);
};

const createBuffer = (size: number): WEBGPUBUFFER => {
  const buffer: WEBGPUBUFFER = {
    size,
    read: async () => readBuffer(buffer)
  };
  states.set(buffer, { size });
  return buffer;
};

export type WEBGPKernel = Kernel & { buffers: WEBGPUBUFFER[] };

const codegenStore = (graph: UOp[], buffers: WEBGPUBUFFER[]): string => {
  const rangeVars = new Map<LowGraph, string>();
  const rangeStack: LowGraph[] = [];
  const lines: string[] = [];
  const constScalar = (u: LowGraph & { op: "CONST" }): number => u.val[0] ?? 0;
  const containsRand = (u: UOp): boolean =>
    u.op === "RAND" || u.srcs.some((s) => containsRand(s));

  const emitIndexExpr = (u: LowGraph): string => {
    if (u.op === "CONST") return `${constScalar(u)}u`;
    if (u.op === "RANGE") {
      const v = rangeVars.get(u);
      if (!v) throw new Error("range used outside scope");
      return v;
    }
    if (u.op === "ADD" || u.op === "MUL") {
      const a = emitIndexExpr(u.srcs[0] as LowGraph);
      const b = emitIndexExpr(u.srcs[1] as LowGraph);
      return `(${a} ${u.op === "ADD" ? "+" : "*"} ${b})`;
    }
    throw new Error(`unsupported index arg: ${u.op}`);
  };

  const emitValueExpr = (u: LowGraph): string => {
    if (u.op === "CONST") {
      const v = constScalar(u);
      return Number.isInteger(v) ? `${v}.0` : String(v);
    }
    if (u.op === "RANGE") {
      const v = rangeVars.get(u);
      if (!v) throw new Error("range used outside scope");
      return `f32(${v})`;
    }
    if (u.op === "ADD" || u.op === "MUL") {
      const a = emitValueExpr(u.srcs[0] as LowGraph);
      const b = emitValueExpr(u.srcs[1] as LowGraph);
      return `(${a} ${u.op === "ADD" ? "+" : "*"} ${b})`;
    }
    if (u.op === "INDEX") {
      const base = u.srcs[0] as LowGraph;
      const idx = u.srcs[1] as LowGraph;
      if (base.op === "RAND") {
        const seed = (Math.floor(base.seed) >>> 0);
        return `randf(${seed}u ^ ${emitIndexExpr(idx)})`;
      }
      if (base.op !== "BUFFER") throw new Error(`unsupported value index base: ${base.op}`);
      const binding = buffers.findIndex((b) => b === (base.buf as WEBGPUBUFFER));
      if (binding < 0) throw new Error("graph references unknown kernel buffer");
      return `b${binding}[${emitIndexExpr(idx)}]`;
    }
    throw new Error(`unsupported store src: ${u.op}`);
  };

  for (const g of graph) {
    if (g.op === "RANGE") {
      const v = `r${rangeVars.size}`;
      rangeVars.set(g, v);
      rangeStack.push(g);
      lines.push(`for (var ${v}:u32 = 0u; ${v} < ${g.max}u; ${v} = ${v} + 1u) {`);
      continue;
    }
    if (g.op === "ENDRANGE") {
      const target = g.srcs[0] as LowGraph;
      const top = rangeStack.pop();
      if (!top || top !== target) throw new Error("ENDRANGE mismatch");
      lines.push("}");
      continue;
    }
    if (g.op !== "STORE") throw new Error(`unsupported root op: ${g.op}`);
    const src = g.srcs[0] as LowGraph;
    const dst = g.srcs[1] as LowGraph;
    if (dst.op !== "INDEX") throw new Error(`unsupported store dst: ${dst.op}`);
    const base = dst.srcs[0] as LowGraph;
    const idx = dst.srcs[1] as LowGraph;
    if (base.op !== "BUFFER") throw new Error(`unsupported index base: ${base.op}`);
    const binding = buffers.findIndex((b) => b === (base.buf as WEBGPUBUFFER));
    if (binding < 0) throw new Error("graph references unknown kernel buffer");
    lines.push(`b${binding}[${emitIndexExpr(idx)}] = ${emitValueExpr(src)};`);
  }
  if (rangeStack.length) throw new Error("unclosed RANGE");
  const needsRand = graph.some((g) => containsRand(g));

  return [
    ...buffers.map((_, i) => `@group(0) @binding(${i}) var<storage, read_write> b${i}: array<f32>;`),
    ...(needsRand ? [
      "fn randf(x:u32) -> f32 {",
      "  var z = x + 0x9e3779b9u;",
      "  z = (z ^ (z >> 16u)) * 0x85ebca6bu;",
      "  z = (z ^ (z >> 13u)) * 0xc2b2ae35u;",
      "  z = z ^ (z >> 16u);",
      "  return f32(z) * 2.3283064365386963e-10;",
      "}"
    ] : []),
    "@compute @workgroup_size(1)",
    "fn main() {",
    ...lines.map((l) => `  ${l}`),
    "}"
  ].join("\n");
};

export const WEBGPU: BACKEND<WEBGPUBUFFER> = {
  createBuffer,
  createKernel: (graph: UOp[], buffers: WEBGPUBUFFER[]) => {
    const wgsl = codegenStore(graph, buffers);
    if (DEBUG.get() === 1) console.log(wgsl);

    const launch = async () => {
      const device = await getDevice();
      const gpuBuffers = await Promise.all(buffers.map(ensureGPUBuffer));

      const module = device.createShaderModule({ code: wgsl });
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: gpuBuffers.map((buffer, i) => ({ binding: i, resource: { buffer } }))
      });

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1, 1, 1);
      pass.end();
      device.queue.submit([enc.finish()]);
    };

    return {
      graph,
      buffers,
      launch
    };
  }
};
