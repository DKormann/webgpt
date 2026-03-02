import { Tensor, BACKEND } from "./tensor.ts";
import { vectorAdd, webgpuAvailable } from "./runtime/webgpu.ts";
import { renderTemplate } from "./template.ts";

const STORAGE_KEY = "playground_code_v1";
const DEFAULT_CODE = `const t = Tensor.new([[1,2,3],[4,5,6]]);
const sum1 = await t.sum([1]).run("webgpu");
const prod0 = await t.prod([0]).run("webgpu");
return { webgpu: webgpuAvailable, sum1, prod0 };`;
const state: Record<string, unknown> = {};
type ScriptCtx = { Tensor: typeof Tensor; BACKEND: typeof BACKEND; webgpuAvailable: boolean };
type ScriptMod = { main: (ctx: ScriptCtx) => Promise<unknown> | unknown };
type ScriptDef = { label: string; module: () => Promise<ScriptMod>; sourcePath: string };

const SCRIPTS: Record<string, ScriptDef> = {
  hello: {
    label: "Hello Script",
    module: () => import("./scripts/hello.ts"),
    sourcePath: "/src/scripts/hello.ts"
  }
};

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
  const def = SCRIPTS[name];
  app.innerHTML = `
    <h1>Script: /${name}</h1>
    <p>${def ? def.label : "Unknown script"}</p>
    <p><button id="script-run">Run</button> <button id="script-code">Show code</button></p>
    <pre id="script-result"></pre>
    <pre id="script-source" style="display:none"></pre>
    <p><a href="./">Back to playground</a></p>
  `;
  const out = app.querySelector<HTMLPreElement>("#script-result");
  const runBtn = app.querySelector<HTMLButtonElement>("#script-run");
  const codeBtn = app.querySelector<HTMLButtonElement>("#script-code");
  const source = app.querySelector<HTMLPreElement>("#script-source");
  if (!out || !runBtn || !codeBtn || !source) return;

  const run = async () => {
    out.textContent = "Running...";
    out.className = "";
    try {
      if (!def) {
        out.textContent = `No script handler for "${name}"`;
        return;
      }
      const mod = await def.module();
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

  runBtn.addEventListener("click", () => void run());
  codeBtn.addEventListener("click", async () => {
    const visible = source.style.display !== "none";
    if (visible) {
      source.style.display = "none";
      codeBtn.textContent = "Show code";
      return;
    }
    if (!def) {
      source.className = "error";
      out.textContent = `No script handler for "${name}"`;
      return;
    }
    if (!source.textContent) {
      try {
        const res = await fetch(def.sourcePath);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        source.className = "";
        source.textContent = await res.text();
      } catch (err) {
        source.className = "error";
        source.textContent = err instanceof Error ? err.message : String(err);
      }
    }
    source.style.display = "block";
    codeBtn.textContent = "Hide code";
  });
  void run();
};

const renderMain = () => {
  app.innerHTML = `
    <h1>WebGPT</h1>
    <p id="status"></p>
    <p><a href="./template">Tensor Template</a> | ${Object.entries(SCRIPTS)
      .map(([name, def]) => `<a href="./${name}">${def.label}</a>`)
      .join(" | ")}</p>
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
  const name = path.slice(1);
  if (name && SCRIPTS[name]) return void runNamedScript(name);

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
