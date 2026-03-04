import { existsSync, readFileSync, readdirSync } from "node:fs";
import { extname, join, normalize } from "node:path";

const cwd = process.cwd();
const port = Number(process.env.PORT ?? 8000);
const scriptsDir = join(cwd, "src", "scripts");
const transpiler = new Bun.Transpiler({ loader: "ts", target: "browser" });

const listScripts = (): string[] =>
  readdirSync(scriptsDir)
    .filter((name) => extname(name) === ".ts")
    .map((name) => name.replace(/\.ts$/, ""))
    .sort((a, b) => a.localeCompare(b));

const htmlShell = (title: string, body: string): string => `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${title}</title>
    <link rel="stylesheet" href="/src/style.css" />
  </head>
  <body>
    ${body}
  </body>
</html>`;

const indexPage = (): string => {
  const scripts = listScripts();
  const rows = scripts
    .map((name) => `<li><a href="/scripts/${name}/">${name}</a> <a href="/scripts/${name}/edit/">edit</a></li>`)
    .join("\n");

  return htmlShell(
    "WebGPT scripts",
    `<main id="app">
      <header class="page-header">
        <h1>Script Pages</h1>
        <p class="status">Each script has a run page and an edit page.</p>
      </header>
      <section class="panel">
        <ul>${rows}</ul>
      </section>
    </main>`
  );
};

const scriptPage = (name: string, mode: "run" | "edit"): string => {
  const basePath = `/scripts/${name}/`;
  const editPath = `${basePath}edit/`;

  return htmlShell(
    `${name} (${mode})`,
    `<meta name="webgpt-script" content="${name}" />
<meta name="webgpt-mode" content="${mode}" />
<meta name="webgpt-source-url" content="/src/scripts/${name}.ts" />
<meta name="webgpt-home-url" content="/" />
<meta name="webgpt-run-url" content="${basePath}" />
<meta name="webgpt-edit-url" content="${editPath}" />
<main id="app"></main>
<script type="module" src="/src/main.ts"></script>`
  );
};

const toFsPath = (pathname: string): string | null => {
  const clean = pathname.startsWith("/") ? pathname.slice(1) : pathname;
  const resolved = normalize(join(cwd, clean));
  if (!resolved.startsWith(cwd)) return null;
  return resolved;
};

const resolveSrcModule = (pathname: string): string | null => {
  const direct = toFsPath(pathname);
  if (direct && existsSync(direct)) return direct;
  if (!extname(pathname)) {
    const tsPath = toFsPath(`${pathname}.ts`);
    if (tsPath && existsSync(tsPath)) return tsPath;
  }
  return null;
};

const server = Bun.serve({
  port,
  fetch(req) {
    const url = new URL(req.url);
    const pathname = url.pathname;

    if (pathname === "/") {
      return new Response(indexPage(), { headers: { "Content-Type": "text/html; charset=utf-8" } });
    }

    if (pathname.startsWith("/scripts/")) {
      const parts = pathname.split("/").filter(Boolean);
      const name = parts[1] ?? "";
      const mode = parts[2] === "edit" ? "edit" : "run";
      const file = join(scriptsDir, `${name}.ts`);
      if (!existsSync(file)) return new Response("Not found", { status: 404 });

      return new Response(scriptPage(name, mode), {
        headers: { "Content-Type": "text/html; charset=utf-8" }
      });
    }

    if (pathname.startsWith("/src/")) {
      const resolvedSrcPath = resolveSrcModule(pathname);
      if (!resolvedSrcPath) return new Response("Not found", { status: 404 });
      if (resolvedSrcPath.endsWith(".ts")) {
        const source = readFileSync(resolvedSrcPath, "utf8");
        return new Response(transpiler.transformSync(source), {
          headers: { "Content-Type": "application/javascript; charset=utf-8" }
        });
      }
      return new Response(Bun.file(resolvedSrcPath));
    }

    const fsPath = toFsPath(pathname);
    if (!fsPath || !existsSync(fsPath)) return new Response("Not found", { status: 404 });

    if (pathname.endsWith(".ts")) {
      const source = readFileSync(fsPath, "utf8");
      return new Response(transpiler.transformSync(source), {
        headers: { "Content-Type": "application/javascript; charset=utf-8" }
      });
    }

    return new Response(Bun.file(fsPath));
  }
});

console.log(`Dev server: http://localhost:${server.port}`);
