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
  const nested = normalize(join(docsDir, clean, "index.html"));
  if (nested.startsWith(docsDir) && existsSync(nested)) return nested;
  return null;
};

const server = Bun.serve({
  port,
  fetch(req) {
    const url = new URL(req.url);
    const fsPath = toFsPath(url.pathname);
    if (!fsPath) return new Response("Not found", { status: 404 });
    return new Response(Bun.file(fsPath));
  }
});

console.log(`Preview: http://localhost:${server.port}`);
