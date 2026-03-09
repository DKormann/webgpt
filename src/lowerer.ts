import { stridesFor } from "./helpers";
import { mkBuffer, mkUop, type Kernel, type UOp, type UOpKind, type View } from "./types";
import { uop } from "./uops";

type Range = UOpKind<"RANGE">
let nextRangeId = 1
const mkRange = (max:number):Range => mkUop("RANGE", [], {id: nextRangeId++, max})


const indexView = (v: View[], buf: UOp, rngs:Range[]) : UOp =>{

  if (v[0].dims.length != rngs.length) throw new Error ("VIEW missmatch")
  const view = v[0];

  let idx: UOp = mkUop("CONST", [], {val:[0]});
  for (let i = 0; i < rngs.length; i++) {
    const r = rngs[i];
    const s = view.strides[i] ?? 0;
    const term = s === 1 ? r : mkUop("MUL", [r, mkUop("CONST", [], {val:[s]})]);
    idx = i === 0 ? term : mkUop("ADD", [idx, term]);
  }

  return mkUop("INDEX", [buf, idx]);
}

export const lowerer = (graph: Kernel): UOpKind<"KERNEL"> =>{
  let rangify = (u:UOp, rngs: Range[] | null = null) : [UOp, Range[]] =>{
    if (u.op == "REDUCE_AXIS"){
      let {axis, srcs: [ch], bin} = u;
      let [c,shp] = rangify(ch)
      let keep = shp.filter((x,i)=>!axis.includes(i))
      return [{
        op: "REDUCE",
        srcs: [c],
        bin,
        keep:keep.map(k=>k.id),
      } , keep]
    }

    if (u.op == "VIEW"){
      let {views : [v0], srcs:[s]} = u;
      if (rngs == null){
        rngs = v0.dims.map(mkRange)
      }
      return [indexView(u.views, s as UOp & {size:number}, rngs),rngs]
    }

    if (u.op =="BUFFER" || u.op == "RAND"){
      let size = u.arg.size;
      if (rngs == null) rngs = [mkRange(size)]
      if (rngs.length > 1 || rngs[0].max != size) throw new Error("wrong ranges")
      return  [u.op == "RAND" ? {...u, srcs: [rngs[0] ] } : mkUop("INDEX", [u, rngs[0]]), rngs]
    }


    if (u.op== "ADD" || u.op == "MUL") {
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
      let dims = rs.map(r=>r.max)
      let st = mkUop("STORE", [k, indexView([{dims, "strides": stridesFor(dims)}], mkBuffer(u.arg.size), rs)])
      u = {...u, srcs:[st]}
    }
    return {...u, srcs:u.srcs.map(go)} as UOp
  }
  
  return go(graph) as UOpKind<"KERNEL">

}
