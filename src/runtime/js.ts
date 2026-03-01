import type { Shape, UOP } from "../uops.ts";
import type { RuntimeExec } from "./types.ts";

type Compiled = (shape: Shape) => number[];

const scalar = (value: number): string => String(value);

const generateCode = (uop: UOP): string => {
  const constIds = new Map<UOP, string>();
  const decls: string[] = [];
  const shapeIds = new Map<string, string>();
  const arrIds = new Map<string, string>();

  const shapeId = (s: Shape): string => {
    const key = JSON.stringify(s);
    const found = shapeIds.get(key);
    if (found) return found;
    const id = `s${shapeIds.size}`;
    shapeIds.set(key, id);
    decls.push(`const ${id}=${key};`);
    return id;
  };
  const arrId = (a: number[]): string => {
    const key = a.join(",");
    const found = arrIds.get(key);
    if (found) return found;
    const id = `a${arrIds.size}`;
    arrIds.set(key, id);
    decls.push(`const ${id}=[${key}];`);
    return id;
  };

  const emitAt = (node: UOP, at: string, sh: string): string => {
    if (node.op === "CONST") {
      const len = node.data.length;
      const one = scalar(node.data[0] ?? 0);
      if (len <= 1) return `(valid(${at},${sh})?${one}:0)`;
      let id = constIds.get(node);
      if (!id) {
        id = `c${constIds.size}`;
        constIds.set(node, id);
        decls.push(`const ${id}=[${node.data.map(scalar).join(",")}];`);
      }
      return `(valid(${at},${sh})?${id}[ridx(${at},${sh},${len})]:0)`;
    }
    if (node.op === "RANGE") return at;
    if (node.op === "REDUCE") {
      const sid = shapeId(node.inShape);
      const did = arrId(node.dims);
      const inner = emitAt(node.src, `mix(${at},r,${sh},${sid},${did})`, sid);
      const rnum = node.dims.map((d) => node.inShape.dims[d]).reduce((a, c) => a * c, 1);
      if (node.bin === "ADD") {
        return `(()=>{let a=0;for(let r=0;r<${rnum};r++)a+=${inner};return a;})()`;
      }
      return `(()=>{let a=1;for(let r=0;r<${rnum};r++)a*=${inner};return a;})()`;
    }
    const a = emitAt(node.srcs[0], at, shapeId(node.srcShapes[0]));
    const b = emitAt(node.srcs[1], at, shapeId(node.srcShapes[1]));
    if (node.op === "ADD") return `(${a}+${b})`;
    if (node.op === "MUL") return `(${a}*${b})`;
    return emitAt(node.srcs[0], b, sh);
  };

  const expr = emitAt(uop, "i", "shape");
  return [
    '"use strict";',
    ...decls,
    "const idx=(i,s)=>{",
    "  let o=s.offset||0;",
    "  for(let d=s.dims.length-1;d>=0;d--){",
    "    const c=i%s.dims[d];",
    "    i=(i/s.dims[d])|0;",
    "    o+=c*s.strides[d];",
    "  }",
    "  return o;",
    "};",
    "const valid=(i,s)=>{",
    "  if(!s.mask) return true;",
    "  for(let d=s.dims.length-1;d>=0;d--){",
    "    const c=i%s.dims[d];",
    "    i=(i/s.dims[d])|0;",
    "    const m=s.mask[d];",
    "    if(c<m[0]||c>=m[1]) return false;",
    "  }",
    "  return true;",
    "};",
    "const ridx=(i,s,l)=>{const j=idx(i,s);return ((j%l)+l)%l;};",
    "const has=(a,v)=>{for(let i=0;i<a.length;i++) if(a[i]===v) return true; return false;};",
    "const mix=(oi,ri,os,is,rd)=>{",
    "  let li=0,m=1,od=os.dims.length-1;",
    "  for(let d=is.dims.length-1;d>=0;d--){",
    "    let c;",
    "    if(has(rd,d)){ c=ri%is.dims[d]; ri=(ri/is.dims[d])|0; }",
    "    else { c=oi%os.dims[od]; oi=(oi/os.dims[od])|0; od--; }",
    "    li+=c*m; m*=is.dims[d];",
    "  }",
    "  return li;",
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

export const exec: RuntimeExec = (uop, shape) => compile(uop)(shape);
