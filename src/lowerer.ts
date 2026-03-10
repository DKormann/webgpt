import { stridesFor } from "./helpers";
import { mkBuffer, mkUop, type Kernel, type UOp, type UOpKind, type View } from "./types";
import { uop } from "./uops";

type Range = UOpKind<"RANGE">
let nextRangeId = 1
const mkRange = (max:number):Range => mkUop("RANGE", [], {id: nextRangeId++, max})
const c = (x:number) => mkUop("CONST", [], [x])


const indexView = (v: View[], buf: UOp, rngs:Range[]) : UOp =>{
  if (v[0].dims.length != rngs.length) throw new Error ("VIEW missmatch")

  let idxs: UOp[] = [...rngs]
  let flat: UOp = mkUop("MUL", [rngs[0]!, c(0)], undefined)

  const mkFlat = (view: View, ls: UOp[]) => {
    let idx: UOp | null = null
    for (let i = 0; i < ls.length; i++) {
      const s = view.strides[i] ?? 0
      if (s === 0) continue
      const term = s === 1 ? ls[i] : mkUop("MUL", [ls[i], c(s)], undefined)
      idx = idx == null ? term : mkUop("ADD", [idx, term], undefined)
    }
    return idx ?? mkUop("MUL", [ls[0]!, c(0)], undefined)
  }

  for (let i = 0; i < v.length; i++) {
    const view = v[i]
    if (view.dims.length != idxs.length) throw new Error("VIEW rank mismatch in stack")
    flat = mkFlat(view, idxs)
    if (i + 1 == v.length) break
    const nd = v[i + 1]!.dims
    const ns = stridesFor(nd)
    idxs = nd.map((d, j) => {
      const q = ns[j] === 1 ? flat : mkUop("DIV", [flat, c(ns[j]!)], undefined)
      return d === 1 ? mkUop("MUL", [q, c(0)], undefined) : mkUop("MOD", [q, c(d)], undefined)
    })
  }

  return mkUop("INDEX", [buf, flat], undefined);
}

export const lowerer = (graph: Kernel): UOpKind<"KERNEL"> =>{
  let rangify = (u:UOp, rngs: Range[] | null = null) : [UOp, Range[]] =>{
    if (u.op == "REDUCE_AXIS"){
      let {arg:{axis, bin}, srcs: [ch]} = u;
      let [c,shp] = rangify(ch)
      let keep = shp.filter((x,i)=>!axis.includes(i))
      return [mkUop("REDUCE", [c], {bin, keep:keep.map(k=>k.arg.id)}) , keep]
    }

    if (u.op == "VIEW"){
      let {arg:{views : [v0]}, srcs:[s]} = u;
      if (rngs == null){
        rngs = v0.dims.map(mkRange)
      }
      return [indexView(u.arg.views, s as UOp & {size:number}, rngs),rngs]
    }

    if (u.op =="BUFFER" || u.op == "RAND"){
      let size = u.arg.size;
      if (rngs == null) rngs = [mkRange(size)]
      if (rngs.length > 1 || rngs[0].arg.max != size) throw new Error("wrong ranges")
      return  [u.op == "RAND" ? {...u, srcs: [rngs[0] ] } : mkUop("INDEX", [u, rngs[0]], undefined), rngs]
    }


    if (u.op== "ADD" || u.op == "MUL" || u.op == "DIV" || u.op == "MOD") {
      let {srcs: [a,b]} = u
      let [a_,shp] = rangify(a)
      let [b_,_] = rangify(b, shp)
      return [{...u, srcs:[a_,b_]}, shp]
    }
    throw new Error("unexpected:"+ uop.fmt(u))
  }

  let go = (u:UOp):UOp => {
    if (u.op == "KERNEL"){
      let {srcs:[c]} = u;
      let [k,rs] = rangify(c)
      let dims = rs.map(r=>r.arg.max)
      let st = mkUop("STORE", [k, indexView([{dims, "strides": stridesFor(dims)}], mkBuffer(u.arg.size), rs)], undefined)
      u = {...u, srcs:[st]}
    }
    return {...u, srcs:u.srcs.map(go)} as UOp
  }
  
  return go(graph) as UOpKind<"KERNEL">

}
