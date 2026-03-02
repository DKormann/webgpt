import { existsSync, readFileSync, watch } from "node:fs";
import { join, normalize } from "node:path";

const cwd = process.cwd();
const port = Number(process.env.PORT ?? 3000);
const watchedPaths = [join(cwd, "index.html"), join(cwd, "src")];
const clients = new Set<(payload: string) => void>();



const toFsPath = (pathname: string): string | null => {
  const clean = pathname.startsWith("/") ? pathname.slice(1) : pathname;
  const resolved = normalize(join(cwd, clean));
  if (!resolved.startsWith(cwd)) return null;
  return resolved;
};

const transpiler = new Bun.Transpiler({ loader: "ts", target: "browser" });

const server = Bun.serve({
  port,
  fetch(req) {
    const url = new URL(req.url);
    const pathname = url.pathname;

    if (pathname === "/") {
      const html = readFileSync(join(cwd, "index.html"), "utf8")
      return new Response(html, { headers: { "Content-Type": "text/html; charset=utf-8" } });
    }

    if (!pathname.includes(".")) {
      if (pathname.startsWith("/src/")) {
        return new Response("Not found", { status: 404 });
      }
      const html = readFileSync(join(cwd, "index.html"), "utf8")
      return new Response(html, { headers: { "Content-Type": "text/html; charset=utf-8" } });
    }

    const fsPath = toFsPath(pathname);
    if (!fsPath) return new Response("Not found", { status: 404 });
    if (!existsSync(fsPath)) return new Response("Not found", { status: 404 });

    if (pathname.endsWith(".ts")) {
      const source = readFileSync(fsPath, "utf8");
      const code = transpiler.transformSync(source);
      return new Response(code, {
        headers: { "Content-Type": "application/javascript; charset=utf-8" }
      });
    }

    return new Response(Bun.file(fsPath));
  }
});

console.log(`Dev server: http://localhost:${server.port}`);
