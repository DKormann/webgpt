import type { UOP } from "./uop.ts";

type BufferLike = ArrayLike<number>;
type Compiled = (outSize: number, buffers: BufferLike[]) => number[];

const scalar = (value: number): string =>
  Number.isFinite(value) ? String(value) : "0";

const toposort = (root: UOP): UOP[] => {
  const seen = new Set<UOP>();
  const nodes: UOP[] = [];
  const visit = (node: UOP): void => {
    if (seen.has(node)) return;
    seen.add(node);
    if ("srcs" in node) {
      for (const src of node.srcs) visit(src);
    }
    nodes.push(node);
  };
  visit(root);
  return nodes;
};

const generateCode = (uop: UOP): string => {
  const nodes = toposort(uop);
  const ids = new Map<UOP, string>();
  nodes.forEach((node, i) => ids.set(node, `n${i}`));

  const lines: string[] = [
    '"use strict";',
    "const out = Math.max(0, outSize | 0);",
    "const safe = (arr, i) => {",
    "  if (!arr) return 0;",
    "  const v = arr[i];",
    "  return Number.isFinite(v) ? v : 0;",
    "};",
    "const range = (n) => {",
    "  const r = new Array(n);",
    "  for (let i = 0; i < n; i++) r[i] = i;",
    "  return r;",
    "};"
  ];

  for (const node of nodes) {
    const id = ids.get(node)!;

    if (node.op === "CONST") {
      const values = node.data.map(scalar).join(", ");
      const len = Math.max(node.data.length, 1);
      lines.push(`const ${id} = new Array(out);`);
      lines.push(`for (let i = 0; i < out; i++) ${id}[i] = [${values}][i % ${len}];`);
      continue;
    }

    if (node.op === "BUFFER") {
      const idx = node.idx | 0;
      lines.push(`const ${id} = new Array(out);`);
      lines.push(
        `for (let i = 0; i < out; i++) ${id}[i] = safe(buffers[${idx}], i);`
      );
      continue;
    }

    if (node.op === "RANGE") {
      lines.push(`const ${id} = range(out);`);
      continue;
    }

    const a = ids.get(node.srcs[0])!;
    const b = ids.get(node.srcs[1])!;
    lines.push(`const ${id} = new Array(out);`);
    if (node.op === "ADD") {
      lines.push(`for (let i = 0; i < out; i++) ${id}[i] = ${a}[i] + ${b}[i];`);
      continue;
    }
    if (node.op === "MUL") {
      lines.push(`for (let i = 0; i < out; i++) ${id}[i] = ${a}[i] * ${b}[i];`);
      continue;
    }
    lines.push(
      `for (let i = 0; i < out; i++) ${id}[i] = safe(${a}, Math.trunc(${b}[i]));`
    );
  }

  lines.push(`return ${ids.get(uop)};`);
  return lines.join("\n");
};

const parseExecArgs = (args: unknown[]): { outSize: number; buffers: BufferLike[] } => {
  if (args.length === 0) return { outSize: 1, buffers: [] };
  if (typeof args[0] === "number") {
    return {
      outSize: Math.max(0, Math.trunc(args[0])),
      buffers: args.slice(1).filter(Array.isArray) as BufferLike[]
    };
  }
  const buffers = args.filter(Array.isArray) as BufferLike[];
  const inferred = buffers[0]?.length ?? 1;
  return { outSize: inferred, buffers };
};

export const compile = (uop: UOP): ((...args: unknown[]) => number[]) => {
  const code = generateCode(uop);
  const fn = new Function("outSize", "buffers", code) as Compiled;
  return (...args: unknown[]) => {
    const { outSize, buffers } = parseExecArgs(args);
    return fn(outSize, buffers);
  };
};

export const exec = (uop: UOP, ...args: unknown[]): number[] => compile(uop)(...args);
