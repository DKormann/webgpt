let devicePromise: Promise<GPUDevice> | null = null;

export const webgpuAvailable = (): boolean =>
  typeof navigator !== "undefined" && !!navigator.gpu;

const getDevice = async (): Promise<GPUDevice> => {
  if (devicePromise) return devicePromise;
  devicePromise = (async () => {
    if (!webgpuAvailable()) throw new Error("WebGPU unavailable");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter");
    return adapter.requestDevice();
  })();
  return devicePromise;
};

export const vectorAdd = async (aIn: number[], bIn: number[]): Promise<number[]> => {
  if (aIn.length !== bIn.length) throw new Error("vectorAdd length mismatch");

  const device = await getDevice();
  const n = aIn.length;
  const bytes = n * 4;

  const aData = new ArrayBuffer(bytes);
  const bData = new ArrayBuffer(bytes);
  new Float32Array(aData).set(aIn);
  new Float32Array(bData).set(bIn);

  const aBuf = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  const bBuf = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  const outBuf = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const readBuf = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  device.queue.writeBuffer(aBuf, 0, aData);
  device.queue.writeBuffer(bBuf, 0, bData);

  const module = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> out: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i < arrayLength(&a)) {
          out[i] = a[i] + b[i];
        }
      }
    `
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" }
  });

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
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const copy = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  aBuf.destroy();
  bBuf.destroy();
  outBuf.destroy();
  readBuf.destroy();

  return Array.from(copy);
};
