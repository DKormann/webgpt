import { Tensor, BACKEND } from "./tensor.ts";
import { vectorAdd, webgpuAvailable } from "./runtime/webgpu.ts";
import { renderTemplate } from "./template.ts";

const STORAGE_KEY = "playground_code_v1";
const DEFAULT_CODE = `const t = Tensor.new([[1,2,3],[4,5,6]]);
const sum1 = await t.sum([1]).run("webgpu");
const prod0 = await t.prod([0]).run("webgpu");
return { webgpu: webgpuAvailable, sum1, prod0 };`;
const state: Record<string, unknown> = {};

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("#app not found");

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor as new (
  ...args: string[]
) => (...args: unknown[]) => Promise<unknown>;

const runPlayground = async (code: string, out: HTMLPreElement) => {
  try {
    out.className = "";
    const fn = new AsyncFunction(
      "Tensor",
      "BACKEND",
      "state",
      "vectorAdd",
      "webgpuAvailable",
      `"use strict";\n${code}`
    );
    const value = await fn(Tensor, BACKEND, state, vectorAdd, webgpuAvailable);
    out.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? "undefined";
  } catch (err) {
    out.className = "error";
    out.textContent = err instanceof Error ? err.stack ?? err.message : String(err);
  }
};

const runNamedScript = async (name: string) => {
  app.innerHTML = `
    <h1>Script: /${name}</h1>
    <pre id="script-result">Running...</pre>
    <p><a href="./">Back to playground</a></p>
  `;
  const out = app.querySelector<HTMLPreElement>("#script-result");
  if (!out) return;

  try {
    const mod =
      name === "hello"
        ? await import("./scripts/hello.ts")
        : null;
    if (!mod || typeof mod.main !== "function") {
      out.textContent = `No script handler for "${name}"`;
      return;
    }
    const value = await mod.main({ Tensor, BACKEND, webgpuAvailable });
    out.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  } catch (err) {
    out.className = "error";
    out.textContent = err instanceof Error ? err.stack ?? err.message : String(err);
  }
};

const renderMain = () => {
  app.innerHTML = `
    <h1>WebGPT</h1>
    <p id="status"></p>
    <p><a href="./template">Tensor Template</a> | <a href="./hello">Hello Script</a></p>
    <textarea id="code" spellcheck="false"></textarea>
    <button id="run">Run</button>
    <pre id="result"></pre>
  `;

  const statusEl = app.querySelector<HTMLParagraphElement>("#status");
  const codeEl = app.querySelector<HTMLTextAreaElement>("#code");
  const runEl = app.querySelector<HTMLButtonElement>("#run");
  const resultEl = app.querySelector<HTMLPreElement>("#result");
  if (!statusEl || !codeEl || !runEl || !resultEl) throw new Error("playground elements missing");

  statusEl.textContent = webgpuAvailable
    ? "WebGPU available. Run with Cmd/Ctrl+Enter."
    : "WebGPU unavailable in this browser";

  codeEl.value = localStorage.getItem(STORAGE_KEY) ?? DEFAULT_CODE;
  codeEl.addEventListener("input", () => localStorage.setItem(STORAGE_KEY, codeEl.value));
  runEl.addEventListener("click", () => void runPlayground(codeEl.value, resultEl));
  codeEl.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      void runPlayground(codeEl.value, resultEl);
    }
  });
  void runPlayground(codeEl.value, resultEl);
};

const normalizePath = (raw: string): string => {
  const path = raw.replace(/\/+$/, "") || "/";
  // Support project pages like /webgpt/... while keeping local root routing.
  if (path === "/webgpt") return "/";
  return path.startsWith("/webgpt/") ? path.slice("/webgpt".length) : path;
};

const renderRoute = () => {
  const path = normalizePath(window.location.pathname);
  if (path === "/" || path === "") return renderMain();
  if (path === "/template") return renderTemplate(app);
  if (path === "/hello") return void runNamedScript("hello");

  app.innerHTML = `<h1>404</h1><p>No route for <code>${path}</code></p><p><a href="./">Back home</a></p>`;
};

document.addEventListener("click", (e) => {
  const target = e.target as HTMLElement | null;
  const anchor = target?.closest("a") as HTMLAnchorElement | null;
  if (!anchor) return;
  const url = new URL(anchor.href);
  if (url.origin !== window.location.origin) return;
  e.preventDefault();
  history.pushState({}, "", url.pathname);
  renderRoute();
});

window.addEventListener("popstate", renderRoute);
renderRoute();

export {};
