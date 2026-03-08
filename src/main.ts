import { DEBUG } from "./debug";
import { Tensor, TensorVar, compile } from "./tensor";

type Mode = "run" | "edit";

type PageConfig = {
  scriptName: string;
  mode: Mode;
  sourceUrl: string;
  homeUrl: string;
  runUrl: string;
  editUrl: string;
};

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("#app not found");

const readMeta = (name: string): string => {
  const node = document.querySelector<HTMLMetaElement>(`meta[name=\"${name}\"]`);
  if (!node?.content) throw new Error(`Missing meta ${name}`);
  return node.content;
};

const config: PageConfig = {
  scriptName: readMeta("webgpt-script"),
  mode: readMeta("webgpt-mode") as Mode,
  sourceUrl: readMeta("webgpt-source-url"),
  homeUrl: readMeta("webgpt-home-url"),
  runUrl: readMeta("webgpt-run-url"),
  editUrl: readMeta("webgpt-edit-url")
};

const storageKey = `webgpt:script:${config.scriptName}`;
const supportsWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;
const BACKEND: undefined = undefined;

const toText = (value: unknown): string => {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean" || value == null) return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return Object.prototype.toString.call(value);
  }
};

const normalizeSource = (source: string): string =>
  source.replace(/^\s*import\s+[^\n]*$/gm, "").replace(/^\s*export\s+/gm, "");

const runUserCode = async (source: string, output: HTMLElement): Promise<void> => {
  const lines: string[] = [];
  const append = (type: "log" | "warn" | "error", values: unknown[]) => {
    const prefix = type === "log" ? "" : `${type.toUpperCase()}: `;
    lines.push(`${prefix}${values.map(toText).join(" ")}`);
    output.textContent = lines.join("\n");
  };

  const captureConsole: Console = {
    ...console,
    log: (...values: unknown[]) => append("log", values),
    warn: (...values: unknown[]) => append("warn", values),
    error: (...values: unknown[]) => append("error", values)
  };

  output.textContent = "Running...";
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor as new (...args: string[]) => (...fnArgs: unknown[]) => Promise<void>;
  const g = globalThis as typeof globalThis & {
    Tensor?: typeof Tensor;
    BACKEND?: undefined;
    DEBUG?: typeof DEBUG;
    TensorVar?: typeof TensorVar;
    compile?: typeof compile;
  };
  const previousTensor = g.Tensor;
  const previousBackend = g.BACKEND;
  const previousDebug = g.DEBUG;
  const previousTensorVar = g.TensorVar;
  const previousCompile = g.compile;

  try {
    g.Tensor = Tensor;
    g.BACKEND = BACKEND;
    g.DEBUG = DEBUG;
    g.TensorVar = TensorVar;
    g.compile = compile;
    const fn = new AsyncFunction("Tensor", "BACKEND", "DEBUG", "TensorVar", "compile", "webgpuAvailable", "console", normalizeSource(source));
    await fn(Tensor, BACKEND, DEBUG, TensorVar, compile, supportsWebGPU, captureConsole);
    if (lines.length === 0) output.textContent = "(no console output)";
  } catch (error) {
    output.classList.add("error");
    append("error", [error instanceof Error ? error.stack ?? error.message : String(error)]);
  } finally {
    g.Tensor = previousTensor;
    g.BACKEND = previousBackend;
    g.DEBUG = previousDebug;
    g.TensorVar = previousTensorVar;
    g.compile = previousCompile;
  }
};

const loadDefaultSource = async (): Promise<string> => {
  const res = await fetch(config.sourceUrl, { cache: "no-cache" });
  if (!res.ok) throw new Error(`Failed loading ${config.sourceUrl}`);
  return res.text();
};

const readCachedSource = (): string | null => {
  try {
    return localStorage.getItem(storageKey);
  } catch {
    return null;
  }
};

const writeCachedSource = (source: string): void => {
  try {
    localStorage.setItem(storageKey, source);
  } catch {
    // ignore storage failures
  }
};

const clearCachedSource = (): void => {
  try {
    localStorage.removeItem(storageKey);
  } catch {
    // ignore storage failures
  }
};

const makeHeader = (): HTMLElement => {
  const header = document.createElement("header");
  header.className = "page-header";
  header.innerHTML = `
    <h1>${config.scriptName}</h1>
    <p class="status">Mode: <strong>${config.mode}</strong> | WebGPU: <strong>${supportsWebGPU ? "available" : "unavailable"}</strong></p>
    <nav class="links">
      <a href="${config.homeUrl}">All scripts</a>
      <a href="${config.runUrl}">Run</a>
      <a href="${config.editUrl}">Edit</a>
    </nav>
  `;
  return header;
};

const mountRunPage = async (source: string): Promise<void> => {
  app.appendChild(makeHeader());
  const panel = document.createElement("section");
  panel.className = "panel";

  const runButton = document.createElement("button");
  runButton.textContent = "Run script";

  const code = document.createElement("pre");
  code.className = "code";
  code.textContent = source;

  const output = document.createElement("pre");
  output.className = "output";

  runButton.addEventListener("click", async () => {
    output.classList.remove("error");
    await runUserCode(source, output);
    code.textContent = source;
  });

  panel.append(runButton, code, output);
  app.appendChild(panel);
  await runUserCode(source, output);
};

const mountEditPage = async (source: string): Promise<void> => {
  app.appendChild(makeHeader());
  const panel = document.createElement("section");
  panel.className = "panel";

  const textarea = document.createElement("textarea");
  textarea.className = "editor";
  textarea.value = readCachedSource() ?? source;

  const actions = document.createElement("div");
  actions.className = "actions";

  const save = document.createElement("button");
  save.textContent = "Save to localStorage";

  const reset = document.createElement("button");
  reset.textContent = "Reset to default";

  const run = document.createElement("button");
  run.textContent = "Run edited script";

  const output = document.createElement("pre");
  output.className = "output";

  const persist = () => writeCachedSource(textarea.value);
  textarea.addEventListener("input", persist);
  save.addEventListener("click", persist);
  reset.addEventListener("click", () => {
    clearCachedSource();
    textarea.value = source;
    output.classList.remove("error");
    output.textContent = "Cache cleared.";
  });
  run.addEventListener("click", async () => {
    output.classList.remove("error");
    persist();
    await runUserCode(textarea.value, output);
  });

  actions.append(save, reset, run);
  panel.append(textarea, actions, output);
  app.appendChild(panel);
  await runUserCode(textarea.value, output);
};

const start = async () => {
  const source = await loadDefaultSource();
  if (config.mode === "edit") return mountEditPage(source);
  return mountRunPage(source);
};

start().catch((error: unknown) => {
  app.innerHTML = `<pre class=\"output error\">${toText(error instanceof Error ? error.stack ?? error.message : error)}</pre>`;
});
