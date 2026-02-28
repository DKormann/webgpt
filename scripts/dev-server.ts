import { existsSync, readFileSync, watch } from "node:fs";
import { join, normalize } from "node:path";

const cwd = process.cwd();
const port = Number(process.env.PORT ?? 3000);
const watchedPaths = [join(cwd, "index.html"), join(cwd, "src")];
const clients = new Set<(payload: string) => void>();

const liveReloadScript = `
<script>
  const es = new EventSource("/__reload");
  es.addEventListener("reload", () => location.reload());
</script>`;

const sendReload = () => {
  for (const push of clients) push("event: reload\\ndata: 1\\n\\n");
};

for (const path of watchedPaths) {
  watch(path, { recursive: true }, () => sendReload());
}

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

    if (pathname === "/__reload") {
      let pushRef: ((payload: string) => void) | null = null;
      const stream = new ReadableStream({
        start(controller) {
          const push = (payload: string) => controller.enqueue(payload);
          pushRef = push;
          clients.add(push);
          push("event: ready\\ndata: 1\\n\\n");
        },
        cancel() {
          if (pushRef) clients.delete(pushRef);
        }
      });
      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive"
        }
      });
    }

    if (pathname === "/") {
      const html = readFileSync(join(cwd, "index.html"), "utf8").replace(
        "</body>",
        `${liveReloadScript}</body>`
      );
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
