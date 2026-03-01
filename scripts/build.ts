import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const cwd = process.cwd();
const docsDir = join(cwd, "docs");
const rawBase = process.env.PUBLIC_BASE ?? "/webgpt";
const base = `/${rawBase.replace(/^\/+|\/+$/g, "")}`;

rmSync(docsDir, { recursive: true, force: true });
mkdirSync(docsDir, { recursive: true });

const result = await Bun.build({
  entrypoints: [join(cwd, "src/main.ts"), join(cwd, "src/template.ts")],
  outdir: docsDir,
  naming: "[name].[ext]",
  target: "browser",
  sourcemap: "none",
  minify: true
});

if (!result.success) {
  for (const log of result.logs) console.error(log);
  process.exit(1);
}

cpSync(join(cwd, "src/style.css"), join(docsDir, "style.css"));

const rewriteHtml = (html: string): string =>
  html
    .replace(/href="\/(?:src\/)?style\.css"/g, `href="${base}/style.css"`)
    .replace(/src="\/(?:src\/)?main\.(?:ts|js)"/g, `src="${base}/main.js"`)
    .replace(/src="\/(?:src\/)?template\.(?:ts|js)"/g, `src="${base}/template.js"`)
    .replace(/href="\/template"/g, `href="${base}/template"`)
    .replace(/href="\/"/g, `href="${base}/"`);

const indexHtml = rewriteHtml(readFileSync(join(cwd, "index.html"), "utf8"));
const templateHtml = rewriteHtml(readFileSync(join(cwd, "template.html"), "utf8"));

writeFileSync(join(docsDir, "index.html"), indexHtml);
writeFileSync(join(docsDir, "template.html"), templateHtml);
mkdirSync(join(docsDir, "template"), { recursive: true });
writeFileSync(join(docsDir, "template", "index.html"), templateHtml);

console.log("Built static site into ./docs");
