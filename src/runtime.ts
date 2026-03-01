import type { Shape, UOP } from "./uops.ts";

type Compiled = (shape: Shape) => number[];

const scalar = (value: number): string => String(value);

const generateCode = (uop: UOP): string => {
  const constIds = new Map<UOP, string>();
  const decls: string[] = [];

  const emitAt = (node: UOP, at: string): string => {
    if (node.op === "CONST") {
      const len = node.data.length;
      if (len <= 1) return scalar(node.data[0] ?? 0);
      let id = constIds.get(node);
      if (!id) {
        id = `c${constIds.size}`;
        constIds.set(node, id);
        decls.push(`const ${id}=[${node.data.map(scalar).join(",")}];`);
      }
      return `${id}[idx(${at},shape)%${len}]`;
    }
    if (node.op === "RANGE") return at;
    const a = emitAt(node.srcs[0], at);
    const b = emitAt(node.srcs[1], at);
    if (node.op === "ADD") return `(${a}+${b})`;
    if (node.op === "MUL") return `(${a}*${b})`;
    return emitAt(node.srcs[0], b);
  };

  const expr = emitAt(uop, "i");
  return [
    '"use strict";',
    ...decls,
    "const idx=(i,s)=>{",
    "  let o=0;",
    "  for(let d=s.dims.length-1;d>=0;d--){",
    "    const c=i%s.dims[d];",
    "    i=(i/s.dims[d])|0;",
    "    o+=c*s.strides[d];",
    "  }",
    "  return o;",
    "};",
    "const out = shape.numel;",
    "const result = new Array(out);",
    `for (let i=0;i<out;i++) result[i]=${expr};`,
    "return result;"
  ].join("\n");
};

export const compile = (uop: UOP): Compiled => {
  const code = generateCode(uop);
  return new Function("shape", code) as Compiled;
};

export const exec = (uop: UOP, shape: Shape): number[] => compile(uop)(shape);
