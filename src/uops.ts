import { stridesFor } from "./helpers";
import { BinOp, DTYPE, mkUop, UOp } from "./types";

export const uop
= {
  fmt: (u:UOp) : string => {
    const counts = new Map<UOp, number>()
    const walk = (x: UOp) => {
      counts.set(x, (counts.get(x) ?? 0) + 1)
      x.srcs.forEach(walk)
    }
    walk(u)

    const shared = new Set<UOp>(Array.from(counts.entries()).filter(([, c]) => c > 1).map(([x]) => x))
    const names = new Map<UOp, string>()
    let next = 0
    shared.forEach((x) => names.set(x, `v${next++}`))

    const emitted = new Set<UOp>()

    const head = (x: UOp): string => {
      let h = x.op
      if (x.arg) Object.entries(x.arg).forEach(([k, v]) => {
        h += ` ${k}:${JSON.stringify(v)}`
      })
      return h
    }

    const render = (x: UOp, indent = ""): string => {
      if (shared.has(x)) {
        const n = names.get(x)!
        if (emitted.has(x)) return `${indent}${n}`
        emitted.add(x)
        const line = `${indent}${n} := ${head(x)}`
        const kids = x.srcs.map((s) => render(s, `${indent}  `)).join("\n")
        return kids ? `${line}\n${kids}` : line
      }
      const line = `${indent}${head(x)}`
      const kids = x.srcs.map((s) => render(s, `${indent}  `)).join("\n")
      return kids ? `${line}\n${kids}` : line
    }

    return render(u)
  },

  const:(val:number[], dtype?:DTYPE )=>{
    if (dtype == undefined) dtype = Number.isInteger(val[0]) ? "int32" : "float32"
    return mkUop("CONST", [], {val, dtype})
  },
  bin: (op: BinOp, ...a:(UOp | number)[]) =>{
    let u = a.map(a=>typeof a == "number" ? uop.const([a]) : a)
    return u.slice(1).reduce((p,c)=> mkUop(op, [p as UOp, c as UOp]), u[0]) as UOp
  },
  add: (...a:(UOp | number)[]) => uop.bin("ADD", ...a),

  mod: (a:(UOp | number), b:(UOp | number), ) => uop.bin("MOD", a,b),
  div: (a:(UOp | number), b:(UOp | number), ) => uop.bin("DIV", a,b),
  idiv: (a:(UOp | number), b:(UOp | number), ) => uop.bin("IDIV", a,b),
  mul: (a:UOp | number,b:UOp | number) => uop.bin("MUL", a,b),
  reshape: (a:UOp, shape: number[]) => mkUop("RESHAPE", [a], {shape}),
  permute: (a:UOp, shape: number[]) => mkUop("PERMUTE", [a], {shape}),
  expand: (a:UOp, shape:number[]) => mkUop("EXPAND", [a], {shape}),
  buff: (size:number, slot:number) => mkUop("BUFFER", [], {size, slot}),
  store: (dest: UOp, src: UOp) => mkUop("STORE",[src,dest]),
  after: (op:UOp, ...deps:UOp[]) => mkUop("AFTER", [op, ...deps]),
  index: (arr:UOp, i:UOp) => mkUop("INDEX", [arr,i]),

  isbinary : (x:UOp) => ["ADD", "MUL", "DIV", "IDIV", "MOD"].includes(x.op),

  reverse: (self:UOp, grad:UOp) =>{
    switch (self.op){
      case "ADD": return [grad, grad]
      case "MUL": return self.srcs.reverse().map(s=>uop.mul(s,grad)) 
      case "RESHAPE": return [uop.reshape(grad, uop.shape(self.srcs[0]))]
      case "PERMUTE": return [uop.permute(grad, uop.shape(self.srcs[0]).map(self.arg.shape.indexOf).map((x,i)=>x==-1?i:x))]
      case "REDUCE_AXIS": if (self.arg.bin == "ADD") return [uop.expand(grad, self.arg.axis)]
      default:
        if (self.srcs.length == 0) return []
        throw new Error("backward not implemented for "+ self.op)
    }
  },

  topo: (u:UOp):UOp[] =>{
    const out: UOp[] = []
    const seen = new Set<UOp>()
    const active = new Set<UOp>()

    const visit = (x: UOp) => {
      if (seen.has(x)) return
      if (active.has(x)) throw new Error("uop.topo cycle detected")
      active.add(x)
      x.srcs.forEach(visit)
      active.delete(x)
      seen.add(x)
      out.push(x)
    }

    visit(u)
    return out
  },

  mpch: (u:UOp, fn:(u:UOp)=>UOp) => ({...u, srcs: u.srcs.map(fn)} as UOp),
  map: (u:UOp, fn:(u:UOp)=>UOp):UOp => fn(uop.mpch(u, x=>uop.map(x, fn))),
  fore: (u:UOp, fn:(u:UOp)=>void) => {fn(u); u.srcs.forEach(x=>uop.fore(x,fn))},

  dedup: (u: UOp): UOp => {
    const memo = new Map<UOp, UOp>()
    const pool = new Map<string, UOp>()
    const ids = new Map<UOp, number>()
    let nextId = 0

    const getId = (x: UOp): number => {
      if (!ids.has(x)) ids.set(x, nextId++)
      return ids.get(x)!
    }

    const canon = (x: UOp): UOp => {
      const hit = memo.get(x)
      if (hit) return hit

      const srcs = x.srcs.map(canon)
      const args = Object.fromEntries(
        Object.entries(x).filter(([k]) => k !== "srcs")
      )

      const key = JSON.stringify({
        op: x.op,
        args,
        srcs: srcs.map(getId),
      })

      const pooled = pool.get(key)
      if (pooled) {
        memo.set(x, pooled)
        return pooled
      }

      const node = { ...x, srcs } as UOp
      pool.set(key, node)
      memo.set(x, node)
      return node
    }

    return canon(u)
  },

  hash: (u: UOp, _opts: { allowBuffers?: boolean } = {}): string => {
    const memo = new Map<UOp, string>()

    const stable = (v: unknown): string => {
      if (v === null || typeof v !== "object") return JSON.stringify(v)
      if (Array.isArray(v)) return `[${v.map(stable).join(",")}]`
      const obj = v as Record<string, unknown>
      const keys = Object.keys(obj).sort()
      return `{${keys.map((k) => `${JSON.stringify(k)}:${stable(obj[k])}`).join(",")}}`
    }

    const fnv1a = (s: string): string => {
      let h = 0x811c9dc5
      for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i)
        h = Math.imul(h, 0x01000193)
      }
      return (h >>> 0).toString(16).padStart(8, "0")
    }

    const go = (x: UOp): string => {
      const hit = memo.get(x)
      if (hit) return hit

      if (x.op === "BUFFER") {
        const out = fnv1a(`BUFFER|slot:${x.arg.slot}|size:${x.arg.size}`)
        memo.set(x, out)
        return out
      }

      const args = Object.fromEntries(
        Object.entries(x).filter(([k]) => k !== "srcs" && k !== "op")
      )
      const srcHashes = x.srcs.map(go)
      const out = fnv1a(`${x.op}|${stable(args)}|[${srcHashes.join(",")}]`)
      memo.set(x, out)
      return out
    }


    return go(u)
  },

  shape: (u: UOp): number[] => {
    if (u.op === "VIEW") return [...u.arg.views[0]!.dims]
    if (u.op === "CONST") return [u.arg.val.length]
    if (u.op === "BUFFER") return [u.arg.size]
    if (u.op === "RAND") return [u.arg.size]
    if (u.op === "RANGE") return [u.arg.max]
    if (u.op === "SPECIAL") return [u.arg.extent]

    if (u.op === "INDEX") return uop.shape(u.srcs[1])

    if (u.op === "ADD" || u.op === "MUL" || u.op === "DIV" || u.op === "MOD") return uop.shape(u.srcs[0])
    if (u.op === "NOOP") return uop.shape(u.srcs[0])
    if (u.op === "STORE") return uop.shape(u.srcs[1])

    if (u.op === "REDUCE_AXIS") return uop.shape(u.srcs[0]).map((d,i)=>u.arg.axis.includes(i) ? 1 : d)

    if (u.op === "RESHAPE" || u.op === "EXPAND" || u.op === "PERMUTE" || u.op === "PAD" || u.op === "SHRINK") {
      throw new Error(`uop.shape: movement op must be rewritten to VIEW first: ${u.op}`)
    }

    if (u.op === "KERNEL" || u.op === "LINEAR" || u.op === "PROGRAMM" || u.op === "DEFINE_REG" || u.op === "ENDRANGE" || u.op === "REDUCE") {
      throw new Error(`uop.shape: unsupported op ${u.op}`)
    }

    throw new Error(`uop.shape: unknown op ${(u as UOp).op}`)
  },
}

export const flattenIndex = (index:UOp[], shape:number[]) =>{
  let strides = stridesFor(shape)
  return uop.add(...index.map((x,i)=>uop.mul(x,strides[i]))) 
}

export const unFlattenIndex = (index:UOp, shape: number[])=>{
  let strides = stridesFor(shape);
  return strides.map((st,i)=> uop.idiv(uop.mod(index, shape[i]*st), st))
}

