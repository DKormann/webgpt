import type { LinearMatmul, Shape, UOP } from "../uops.ts";
import { exec as jsExec } from "./js.ts";
import { linearize } from "./linearize.ts";
import { buildPlan } from "./plan.ts";
import type { RuntimeExecAsync } from "./types.ts";

let devicePromise: Promise<GPUDevice> | null = null;

export const webgpuAvailable =
  typeof navigator !== "undefined" && "gpu" in navigator && !!navigator.gpu;

const getDevice = async (): Promise<GPUDevice> => {
  if (devicePromise) return devicePromise;
  devicePromise = (async () => {
    if (!webgpuAvailable) throw new Error("WebGPU unavailable");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter");
    return adapter.requestDevice();
  })();
  return devicePromise;
};

const f32 = (v: number): string => {
  if (!Number.isFinite(v)) return "0.0";
  if (Object.is(v, -0)) return "-0.0";
  if (Number.isInteger(v)) return `${v}.0`;
  return String(v);
};

type ConstNode = { id: string; data: number[] };

const genKernel = (uop: UOP, outShape: Shape) => {
  const plan = buildPlan(uop);
  const constNodes = new Map<UOP, ConstNode>();
  const shapeFns = new Map<string, { id: string; shape: Shape }>();
  const reduceFns = new Map<UOP, string>();
  const helperFns: string[] = [];
  const reduceImpls: string[] = [];
  let tmp = 0;

  const getShape = (s: Shape) => {
    const key = JSON.stringify(s);
    const found = shapeFns.get(key);
    if (found) return found.id;
    const id = `s${shapeFns.size}`;
    shapeFns.set(key, { id, shape: s });
    return id;
  };

  const shapeImpl = (id: string, s: Shape): string => {
    const rank = s.dims.length;
    const offset = Math.trunc(s.offset ?? 0);
    const idxLines: string[] = [
      `fn idx_${id}(i0:u32) -> i32 {`,
      `  var i:i32 = i32(i0);`,
      `  var o:i32 = ${offset};`
    ];
    for (let d = rank - 1; d >= 0; d--) {
      const dim = Math.max(1, Math.trunc(s.dims[d]));
      const st = Math.trunc(s.strides[d] ?? 0);
      idxLines.push(`  let c${d}:i32 = i % ${dim};`);
      idxLines.push(`  i = i / ${dim};`);
      idxLines.push(`  o = o + c${d} * ${st};`);
    }
    idxLines.push("  return o;");
    idxLines.push("}");

    const validLines: string[] = [
      `fn valid_${id}(i0:u32) -> bool {`,
      `  var i:i32 = i32(i0);`
    ];
    if (!s.mask) {
      validLines.push("  _ = i;");
      validLines.push("  return true;");
      validLines.push("}");
    } else {
      for (let d = rank - 1; d >= 0; d--) {
        const dim = Math.max(1, Math.trunc(s.dims[d]));
        const lo = Math.trunc(s.mask[d]?.[0] ?? 0);
        const hi = Math.trunc(s.mask[d]?.[1] ?? dim);
        validLines.push(`  let c${d}:i32 = i % ${dim};`);
        validLines.push(`  i = i / ${dim};`);
        validLines.push(`  if (c${d} < ${lo} || c${d} >= ${hi}) { return false; }`);
      }
      validLines.push("  return true;");
      validLines.push("}");
    }

    const ridx = [
      `fn ridx_${id}(i:u32, l:u32) -> u32 {`,
      `  let j:i32 = idx_${id}(i);`,
      "  let m:i32 = i32(l);",
      "  return u32(((j % m) + m) % m);",
      "}"
    ];

    return [...idxLines, ...validLines, ...ridx].join("\n");
  };

  const mkMixFn = (node: Extract<UOP, { op: "REDUCE" }>): { name: string; code: string } => {
    const id = `mix_${plan.id(node)}`;
    const rd = new Set(node.dims);
    const outDims = node.inShape.dims.filter((_, i) => !rd.has(i));
    const lines = [
      `fn ${id}(oi0:u32, ri0:u32) -> u32 {`,
      "  var oi:i32 = i32(oi0);",
      "  var ri:i32 = i32(ri0);",
      "  var li:i32 = 0;",
      "  var m:i32 = 1;",
      `  var od:i32 = ${node.inShape.dims.length - node.dims.length - 1};`
    ];
    for (let d = node.inShape.dims.length - 1; d >= 0; d--) {
      const inDim = Math.max(1, Math.trunc(node.inShape.dims[d]));
      if (rd.has(d)) {
        lines.push(`  let c${d}:i32 = ri % ${inDim};`);
        lines.push(`  ri = ri / ${inDim};`);
        lines.push(`  li = li + c${d} * m;`);
      } else {
        const od = (() => {
          let n = 0;
          for (let k = 0; k <= d; k++) if (!rd.has(k)) n++;
          return n - 1;
        })();
        const outDim = Math.max(1, Math.trunc(outDims[od] ?? 1));
        lines.push(`  let dim${d}:i32 = ${outDim};`);
        lines.push(`  let c${d}:i32 = oi % dim${d};`);
        lines.push(`  oi = oi / dim${d};`);
        lines.push(`  li = li + c${d} * m;`);
      }
      lines.push(`  m = m * ${inDim};`);
    }
    lines.push("  return u32(li);");
    lines.push("}");
    return { name: id, code: lines.join("\n") };
  };

  const emitAt = (
    node: UOP,
    at: string,
    shId: string,
    scopeLines: string[],
    scopeMemo: Map<string, string>
  ): string => {
    const key = `${plan.id(node)}|${at}|${shId}`;
    const materialize = node !== uop && plan.refCount(node) > 1;
    if (materialize) {
      const hit = scopeMemo.get(key);
      if (hit) return hit;
    }
    let expr: string;
    if (node.op === "CONST") {
      const len = node.data.length;
      if (len <= 1) expr = `select(0.0, ${f32(node.data[0] ?? 0)}, valid_${shId}(${at}))`;
      else {
        let cn = constNodes.get(node);
        if (!cn) {
          cn = { id: `c${constNodes.size}`, data: node.data };
          constNodes.set(node, cn);
        }
        expr = `select(0.0, ${cn.id}[ridx_${shId}(${at}, ${len}u)], valid_${shId}(${at}))`;
      }
    } else if (node.op === "RANGE") expr = `f32(${at})`;
    else if (node.op === "REDUCE") {
      const hit = reduceFns.get(node);
      if (hit) return `${hit}(${at})`;
      const inSid = getShape(node.inShape);
      const mixFn = mkMixFn(node);
      helperFns.push(mixFn.code);
      const mixName = mixFn.name;
      const reduceScopeLines: string[] = [];
      const reduceScopeMemo = new Map<string, string>();
      const inner = emitAt(node.src, `${mixName}(oi, r)`, inSid, reduceScopeLines, reduceScopeMemo);
      const rnum = node.dims.map((d) => node.inShape.dims[d]).reduce((a, c) => a * c, 1);
      const rid = `red_${plan.id(node)}`;
      reduceFns.set(node, rid);
      reduceImpls.push([
        `fn ${rid}(oi:u32) -> f32 {`,
        `  var a:f32 = ${node.bin === "ADD" ? "0.0" : "1.0"};`,
        "  var r:u32 = 0u;",
        "  loop {",
        `    if (r >= ${Math.max(1, rnum)}u) { break; }`,
        ...reduceScopeLines.map((l) => `    ${l}`),
        `    a = a ${node.bin === "ADD" ? "+" : "*"} ${inner};`,
        "    r = r + 1u;",
        "  }",
        "  return a;",
        "}"
      ].join("\n"));
      expr = `${rid}(${at})`;
    } else {
      const a = emitAt(node.srcs[0], at, getShape(node.srcShapes[0]), scopeLines, scopeMemo);
      const b = emitAt(node.srcs[1], at, getShape(node.srcShapes[1]), scopeLines, scopeMemo);
      expr = node.op === "ADD" ? `(${a}+${b})` : `(${a}*${b})`;
    }
    if (materialize) {
      const name = `v${tmp++}`;
      scopeLines.push(`let ${name}:f32 = ${expr};`);
      scopeMemo.set(key, name);
      return name;
    }
    return expr;
  };

  const outShapeId = getShape(outShape);
  const mainScopeLines: string[] = [];
  const mainScopeMemo = new Map<string, string>();
  const expr = emitAt(uop, "i", outShapeId, mainScopeLines, mainScopeMemo);

  const constDecls = Array.from(constNodes.values()).map(
    (c, i) => `@group(0) @binding(${i}) var<storage, read> ${c.id}: array<f32>;`
  );
  const outBinding = constNodes.size;
  const outDecl = `@group(0) @binding(${outBinding}) var<storage, read_write> out: array<f32>;`;

  const shapeImpls = Array.from(shapeFns.values()).map((x) => shapeImpl(x.id, x.shape));

  const wgsl = [
    ...constDecls,
    outDecl,
    ...shapeImpls,
    ...helperFns,
    ...reduceImpls,
    "@compute @workgroup_size(64)",
    "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
    "  let i:u32 = gid.x;",
    `  if (i >= ${Math.max(1, outShape.numel)}u) { return; }`,
    ...mainScopeLines.map((l) => `  ${l}`),
    `  out[i] = ${expr};`,
    "}"
  ].join("\n");

  return { wgsl, constNodes: Array.from(constNodes.values()), outBinding };
};

const runLinearMatmul = async (p: LinearMatmul): Promise<number[]> => {
  const device = await getDevice();
  const [TM, TN, TK] = p.tile;
  const wgsl = [
    "@group(0) @binding(0) var<storage, read> A: array<f32>;",
    "@group(0) @binding(1) var<storage, read> B: array<f32>;",
    "@group(0) @binding(2) var<storage, read_write> C: array<f32>;",
    `var<workgroup> As: array<array<f32, ${TK}>, ${TM}>;`,
    `var<workgroup> Bs: array<array<f32, ${TN}>, ${TK}>;`,
    `@compute @workgroup_size(${p.workgroup[0]}, ${p.workgroup[1]}, ${p.workgroup[2]})`,
    "fn main(",
    "  @builtin(workgroup_id) wg: vec3<u32>,",
    "  @builtin(local_invocation_id) lid: vec3<u32>",
    ") {",
    `  let row:u32 = wg.y * ${TM}u + lid.y;`,
    `  let col:u32 = wg.x * ${TN}u + lid.x;`,
    "  var acc:f32 = 0.0;",
    "  var k0:u32 = 0u;",
    "  loop {",
    `    if (k0 >= ${p.K}u) { break; }`,
    "    let ka:u32 = k0 + lid.x;",
    "    let kb:u32 = k0 + lid.y;",
    `    if (row < ${p.M}u && ka < ${p.K}u) { As[lid.y][lid.x] = A[row * ${p.K}u + ka]; } else { As[lid.y][lid.x] = 0.0; }`,
    `    if (kb < ${p.K}u && col < ${p.N}u) { Bs[lid.y][lid.x] = B[kb * ${p.N}u + col]; } else { Bs[lid.y][lid.x] = 0.0; }`,
    "    workgroupBarrier();",
    `    for (var k:u32 = 0u; k < ${TK}u; k = k + 1u) {`,
    "      acc = acc + As[lid.y][k] * Bs[k][lid.x];",
    "    }",
    "    workgroupBarrier();",
    `    k0 = k0 + ${TK}u;`,
    "  }",
    `  if (row < ${p.M}u && col < ${p.N}u) { C[row * ${p.N}u + col] = acc; }`,
    "}"
  ].join("\n");

  const a = new Float32Array(p.a.data);
  const b = new Float32Array(p.b.data);
  const outSize = p.M * p.N;
  const outBytes = outSize * 4;
  const aBuf = device.createBuffer({ size: Math.max(4, a.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bBuf = device.createBuffer({ size: Math.max(4, b.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.queue.writeBuffer(aBuf, 0, a.buffer);
  device.queue.writeBuffer(bBuf, 0, b.buffer);

  const module = device.createShaderModule({ code: wgsl });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: outBuf } }
    ]
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(p.N / TN), Math.ceil(p.M / TM), 1);
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes);
  device.queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const copy = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  aBuf.destroy();
  bBuf.destroy();
  outBuf.destroy();
  readBuf.destroy();
  return Array.from(copy).slice(0, outSize);
};

const runKernel = async (uop: UOP, shape: Shape): Promise<number[]> => {
  let device: GPUDevice;
  device = await getDevice();
  const { wgsl, constNodes, outBinding } = genKernel(uop, shape);
  const outSize = Math.max(1, shape.numel);
  const bytes = outSize * 4;

  const constBuffers = constNodes.map((c) => {
    const data = new Float32Array(c.data.length || 1);
    if (c.data.length) data.set(c.data);
    const buf = device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(buf, 0, data.buffer);
    return buf;
  });

  const outBuf = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const readBuf = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  const module = device.createShaderModule({ code: wgsl });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" }
  });

  const entries = [
    ...constBuffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    { binding: outBinding, resource: { buffer: outBuf } }
  ];
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(outSize / 64));
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const copy = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  for (const b of constBuffers) b.destroy();
  outBuf.destroy();
  readBuf.destroy();

  return Array.from(copy).slice(0, shape.numel);
};

export const execAsync: RuntimeExecAsync = async (uop, shape) => {
  if (!webgpuAvailable) return jsExec(uop, shape);
  const p = linearize(uop);
  if (p) return runLinearMatmul(p);

  // Split nested reductions into two kernels so the inner reduction can run in parallel.
  if (uop.op === "REDUCE" && uop.src.op === "REDUCE") {
    const mid = await execAsync(uop.src, uop.inShape);
    const staged: UOP = { ...uop, src: { op: "CONST", data: mid } };
    return runKernel(staged, shape);
  }

  return runKernel(uop, shape);
};

export const vectorAdd = async (aIn: number[], bIn: number[]): Promise<number[]> => {
  if (aIn.length !== bIn.length) throw new Error("vectorAdd length mismatch");
  const sa: Shape = { dims: [aIn.length], strides: [1], numel: aIn.length };
  const sb: Shape = { dims: [bIn.length], strides: [1], numel: bIn.length };
  const two: UOP = {
    op: "ADD",
    srcs: [{ op: "CONST", data: aIn }, { op: "CONST", data: bIn }],
    srcShapes: [sa, sb]
  };
  return execAsync(two, { dims: [aIn.length], strides: [1], numel: aIn.length });
};
