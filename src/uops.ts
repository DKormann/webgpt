import type { BinOp, RAWBUFFER, UOp, View } from "./types";


export const uop
// : Record <string, (...args: any[]) => UOp>
= {

  add: (a:UOp, b:UOp):UOp=> ({op: "ADD", srcs:[a,b]}),
  mul: (a:UOp, b:UOp):UOp=> ({op: "MUL", srcs:[a,b]}),

  buffer : (buf: RAWBUFFER):UOp & {op:"BUFFER"}=> ({
    op:"BUFFER",
    srcs:[],
    buf,
  }),


  range : (max:number):UOp & {op:"RANGE"} => ({op:"RANGE", srcs:[], max}),
  endrange : (range: UOp & {op: "RANGE"}) : UOp & {op:"ENDRANGE"} => ({op:"ENDRANGE", srcs:[range]}),

  const : (...val: number[]): UOp & {op:"CONST"} => ({
    op: "CONST",
    val,
    srcs:[]
  }),

  rand : (seed: number, size?: number):UOp => ({op:"RAND", seed, srcs:[], size}),

  view: (src: UOp, views: View[]): UOp & { op: "VIEW" } => ({
    op: "VIEW",
    srcs: [src],
    views
  }),

  reduce: (src: UOp, axis: number[], bin: BinOp): UOp & { op: "REDUCE_AXIS" } => ({
    op: "REDUCE_AXIS",
    srcs: [src],
    axis,
    bin
  }),


  store :  (src: UOp, dest: UOp, index?:UOp) : UOp & {op:"STORE"} =>({
    op: "STORE",
    srcs: [
      src,
      index ? uop.index(dest, index) : dest
    ]
  }),
  index: (buf: UOp, index: UOp): UOp => ({op:"INDEX", srcs:[buf,index]}),

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
      Object.entries(x).forEach(([k, v]) => {
        if (!["srcs", "op", "seed"].includes(k)) h += ` ${k}:${JSON.stringify(v)}`
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

  topo: (u:UOp):UOp[] =>{
    let done: UOp[] = [];

  }


}
