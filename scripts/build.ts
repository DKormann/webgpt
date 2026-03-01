import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const cwd = process.cwd();
const docsDir = join(cwd, "docs");
const rawBase = process.env.PUBLIC_BASE ?? "/webgpt";
const base = `/${rawBase.replace(/^\/+|\/+$/g, "")}`;

rmSync(docsDir, { recursive: true, force: true });
mkdirSync(docsDir, { recursive: true });

const result = await Bun.build({
  entrypoints: [join(cwd, "src/main.ts")],
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

const html = readFileSync(join(cwd, "index.html"), "utf8")
  .replace(/href="\/(?:src\/)?style\.css"/, `href="${base}/style.css"`)
  .replace(/src="\/(?:src\/)?main\.(?:ts|js)"/, `src="${base}/main.js"`);

writeFileSync(join(docsDir, "index.html"), html);

console.log("Built static site into ./docs");
