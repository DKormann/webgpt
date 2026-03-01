const STORAGE_KEY = "playground_code_v1";
const DEFAULT_CODE = `const a = [1, 2, 3, 4];
const b = [10, 20, 30, 40];
const c = await vectorAdd(a, b);
return { webgpu: webgpuAvailable(), a, b, c };`;
const state: Record<string, unknown> = {};

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("#app not found");

app.innerHTML = `
  <h1>WebGPT</h1>
  <p id="status"></p>
  <textarea id="code" spellcheck="false"></textarea>
  <button id="run">Run</button>
  <pre id="result"></pre>
`;

const statusEl = document.querySelector<HTMLParagraphElement>("#status");
const codeEl = document.querySelector<HTMLTextAreaElement>("#code");
const runEl = document.querySelector<HTMLButtonElement>("#run");
const resultEl = document.querySelector<HTMLPreElement>("#result");

if (!statusEl || !codeEl || !runEl || !resultEl) {
  throw new Error("playground elements missing");
}

const { vectorAdd, webgpuAvailable } = await import("./runtime/webgpu.ts");
const { Tensor, BACKEND } = await import("./tensor.ts");
statusEl.textContent = webgpuAvailable()
  ? "WebGPU available. Run with Cmd/Ctrl+Enter. Persist values in state."
  : "WebGPU unavailable in this browser";

codeEl.value = localStorage.getItem(STORAGE_KEY) ?? DEFAULT_CODE;
codeEl.addEventListener("input", () => localStorage.setItem(STORAGE_KEY, codeEl.value));

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor as new (
  ...args: string[]
) => (...args: unknown[]) => Promise<unknown>;

const runCode = async () => {
  try {
    resultEl.className = "";
    const fn = new AsyncFunction(
      "Tensor",
      "BACKEND",
      "state",
      "vectorAdd",
      "webgpuAvailable",
      `"use strict";\n${codeEl.value}`
    );
    const value = await fn(Tensor, BACKEND, state, vectorAdd, webgpuAvailable);
    resultEl.textContent =
      typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? "undefined";
  } catch (err) {
    resultEl.className = "error";
    resultEl.textContent = err instanceof Error ? err.stack ?? err.message : String(err);
  }
};

runEl.addEventListener("click", () => void runCode());
codeEl.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    void runCode();
  }
});
void runCode();

export {};
