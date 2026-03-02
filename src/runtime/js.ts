import type { Shape, UOP } from "../uops.ts";
import { buildPlan } from "./plan.ts";
import type { RuntimeExec } from "./types.ts";

type Compiled = (shape: Shape) => number[];

const scalar = (value: number): string => String(value);

const generateCode = (uop: UOP): string => {
  const plan = buildPlan(uop);
  const constIds = new Map<UOP, string>();
  const decls: string[] = [];
  const shapeIds = new Map<string, string>();
  const arrIds = new Map<string, string>();
  let tmp = 0;

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

  const emitAt = (
    node: UOP,
    at: string,
    sh: string,
    scopeLines: string[],
    scopeMemo: Map<string, string>
  ): string => {
    const key = `${plan.id(node)}|${at}|${sh}`;
    const materialize = node !== uop && plan.refCount(node) > 1;
    if (materialize) {
      const hit = scopeMemo.get(key);
      if (hit) return hit;
    }

    let expr: string;
    if (node.op === "CONST") {
      const len = node.data.length;
      const one = scalar(node.data[0] ?? 0);
      if (len <= 1) expr = `(valid(${at},${sh})?${one}:0)`;
      else {
        let id = constIds.get(node);
        if (!id) {
          id = `c${constIds.size}`;
          constIds.set(node, id);
          decls.push(`const ${id}=[${node.data.map(scalar).join(",")}];`);
        }
        expr = `(valid(${at},${sh})?${id}[ridx(${at},${sh},${len})]:0)`;
      }
    } else if (node.op === "RAND") {
      expr = `(valid(${at},${sh})?randv(idx(${at},${sh}),${node.seed | 0}):0)`;
    } else if (node.op === "RANGE") expr = at;
    else if (node.op === "REDUCE") {
      const sid = shapeId(node.inShape);
      const did = arrId(node.dims);
      const reduceScopeLines: string[] = [];
      const reduceScopeMemo = new Map<string, string>();
      const inner = emitAt(node.src, `mix(${at},r,${sh},${sid},${did})`, sid, reduceScopeLines, reduceScopeMemo);
      const rnum = node.dims.map((d) => node.inShape.dims[d]).reduce((a, c) => a * c, 1);
      if (node.bin === "ADD") {
        expr = `(()=>{let a=0;for(let r=0;r<${rnum};r++){${reduceScopeLines.join("")}a+=${inner};}return a;})()`;
      } else {
        expr = `(()=>{let a=1;for(let r=0;r<${rnum};r++){${reduceScopeLines.join("")}a*=${inner};}return a;})()`;
      }
    } else {
      const a = emitAt(node.srcs[0], at, shapeId(node.srcShapes[0]), scopeLines, scopeMemo);
      const b = emitAt(node.srcs[1], at, shapeId(node.srcShapes[1]), scopeLines, scopeMemo);
      expr = node.op === "ADD" ? `(${a}+${b})` : `(${a}*${b})`;
    }

    if (materialize) {
      const name = `v${tmp++}`;
      scopeLines.push(`const ${name}=${expr};`);
      scopeMemo.set(key, name);
      return name;
    }
    return expr;
  };

  const loopLines: string[] = [];
  const loopMemo = new Map<string, string>();
  const expr = emitAt(uop, "i", "shape", loopLines, loopMemo);
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
    "const randv=(j,seed)=>{let x=((j|0)^(seed|0)^0x9e3779b9)|0;x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296;};",
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
    "for (let i=0;i<out;i++){",
    ...loopLines.map((l) => `  ${l}`),
    `  result[i]=${expr};`,
    "}",
    "return result;"
  ].join("\n");
};

export const compile = (uop: UOP): Compiled => {
  const code = generateCode(uop);
  return new Function("shape", code) as Compiled;
};

export const exec: RuntimeExec = (uop, shape) => compile(uop)(shape);
