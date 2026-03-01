const app = document.querySelector<HTMLDivElement>("#app");

if (app) {
  app.innerHTML = `<h1>WebGPU Demo</h1><p id="out">Running...</p>`;
  const out = document.querySelector<HTMLParagraphElement>("#out");
  if (out) {
    const { vectorAdd, webgpuAvailable } = await import("./runtime/webgpu.ts");
    if (!webgpuAvailable()) {
      out.textContent = "WebGPU unavailable in this browser.";
    } else {
      try {
        const a = [1, 2, 3, 4];
        const b = [10, 20, 30, 40];
        const c = await vectorAdd(a, b);
        out.textContent = `vectorAdd([${a}], [${b}]) = [${c}]`;
      } catch (err) {
        out.textContent = `WebGPU error: ${String(err)}`;
      }
    }
  }
}

export {};
