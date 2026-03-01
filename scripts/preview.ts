import { existsSync } from "node:fs";
import { join, normalize } from "node:path";

const cwd = process.cwd();
const docsDir = join(cwd, "docs");
const port = Number(process.env.PORT ?? 4173);

const toFsPath = (pathname: string): string | null => {
  const clean = pathname === "/" ? "/index.html" : pathname;
  const resolved = normalize(join(docsDir, clean));
  if (!resolved.startsWith(docsDir)) return null;
  if (existsSync(resolved)) return resolved;
  if (!clean.includes(".")) {
    const htmlResolved = normalize(join(docsDir, `${clean}.html`));
    if (htmlResolved.startsWith(docsDir) && existsSync(htmlResolved)) return htmlResolved;
  }
  return resolved;
};

const server = Bun.serve({
  port,
  fetch(req) {
    const url = new URL(req.url);
    const fsPath = toFsPath(url.pathname);
    if (!fsPath) return new Response("Not found", { status: 404 });
    if (!existsSync(fsPath)) return new Response("Not found", { status: 404 });
    return new Response(Bun.file(fsPath));
  }
});

console.log(`Preview: http://localhost:${server.port}`);
