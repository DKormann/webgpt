import { cpSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { extname, join } from "node:path";

const cwd = process.cwd();
const docsDir = join(cwd, "docs");
const scriptsDir = join(cwd, "src", "scripts");

const listScripts = (): string[] =>
  readdirSync(scriptsDir)
    .filter((name) => extname(name) === ".ts")
    .map((name) => name.replace(/\.ts$/, ""))
    .sort((a, b) => a.localeCompare(b));

const htmlShell = (title: string, cssPath: string, body: string): string => `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${title}</title>
    <link rel="stylesheet" href="${cssPath}" />
  </head>
  <body>
    ${body}
  </body>
</html>`;

const scriptPage = (name: string, mode: "run" | "edit"): string => {
  const isEdit = mode === "edit";
  const assetPrefix = isEdit ? "../../../assets" : "../../assets";
  const sourceUrl = isEdit ? "../../../script-sources" : "../../script-sources";
  const homeUrl = isEdit ? "../../../" : "../../";
  const runUrl = isEdit ? "../" : "./";
  const editUrl = isEdit ? "./" : "edit/";

  return htmlShell(
    `${name} (${mode})`,
    `${assetPrefix}/style.css`,
    `<meta name="webgpt-script" content="${name}" />
<meta name="webgpt-mode" content="${mode}" />
<meta name="webgpt-source-url" content="${sourceUrl}/${name}.ts" />
<meta name="webgpt-home-url" content="${homeUrl}" />
<meta name="webgpt-run-url" content="${runUrl}" />
<meta name="webgpt-edit-url" content="${editUrl}" />
<main id="app"></main>
<script type="module" src="${assetPrefix}/main.js"></script>`
  );
};

const rootPage = (scripts: string[]): string => {
  const rows = scripts
    .map((name) => `<li><a href="./scripts/${name}/">${name}</a> <a href="./scripts/${name}/edit/">edit</a></li>`)
    .join("\n");

  return htmlShell(
    "WebGPT scripts",
    "./assets/style.css",
    `<main id="app">
      <header class="page-header">
        <h1>Script Pages</h1>
        <p class="status">Static build for GitHub Pages.</p>
      </header>
      <section class="panel">
        <ul>${rows}</ul>
      </section>
    </main>`
  );
};

rmSync(docsDir, { recursive: true, force: true });
mkdirSync(join(docsDir, "assets"), { recursive: true });
mkdirSync(join(docsDir, "scripts"), { recursive: true });
mkdirSync(join(docsDir, "script-sources"), { recursive: true });

const result = await Bun.build({
  entrypoints: [join(cwd, "src", "main.ts")],
  outdir: join(docsDir, "assets"),
  naming: "[name].[ext]",
  target: "browser",
  minify: true,
  sourcemap: "none"
});

if (!result.success) {
  for (const log of result.logs) console.error(log);
  process.exit(1);
}

cpSync(join(cwd, "src", "style.css"), join(docsDir, "assets", "style.css"));

const scripts = listScripts();
for (const name of scripts) {
  const runDir = join(docsDir, "scripts", name);
  const editDir = join(runDir, "edit");
  mkdirSync(editDir, { recursive: true });

  writeFileSync(join(runDir, "index.html"), scriptPage(name, "run"));
  writeFileSync(join(editDir, "index.html"), scriptPage(name, "edit"));
  cpSync(join(scriptsDir, `${name}.ts`), join(docsDir, "script-sources", `${name}.ts`));
}

const indexHtml = rootPage(scripts);
writeFileSync(join(docsDir, "index.html"), indexHtml);
writeFileSync(join(docsDir, "404.html"), indexHtml);

console.log(`Built ${scripts.length} script page(s) into ./docs`);
