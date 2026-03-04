import { PatternMatcher, UPat } from "./patter_matcher";
import type { Op, RAWBUFFER, Schedule, UOp, UOpKind, View } from "./types";
import { uop } from "./uops";




let pm = new PatternMatcher([

])

type Range = UOpKind<"RANGE">


const indexView = (v: View[], buf: UOp & {size:number}, rngs:Range[]) : UOp =>{

  if (v[0].dims.length != rngs.length) throw new Error ("VIEW missmatch")

}

export const lowerer = (graph: UOp): UOp =>{

  let rangify = (u:UOp, rngs: Range[] | null = null) : [UOp, Range[]] =>{
    if (u.op == "REDUCE_AXIS"){
      let {axis, srcs: [ch], bin} = u;
      let [c,shp] = rangify(ch)
      let keep = shp.filter((x,i)=>!axis.includes(i))
      return [{
        op: "REDUCE",
        srcs: [c],
        bin,
        keep,
      } , keep]
    }

    if (u.op == "VIEW"){
      let {views : [v0], srcs:[s]} = u;
      if (rngs == null){
        rngs = v0.dims.map(uop.range)
      }
      return [indexView(u.views, s as UOp & {size:number}, rngs),rngs]
    }

    if (u.op =="BUFFER" || u.op =="RAND"){

    }

    if (u.op== "ADD" || u.op == "MUL") {
      let {srcs: [a,b]} = u
      let [a_,shp] = rangify(a)
      let [b_,_] = rangify(b)
      return [{...u, srcs:[a_,b_]}, shp]
    }
    throw new Error("unexpected:"+ uop.fmt(u))
  }

  let go = (u:UOp):UOp => {
    if (u.op == "KERNEL"){
      let {srcs:[c]} = u;
      return {...u, srcs:[go(rangify(c)[0])]}
    }
    return {...u, srcs:u.srcs.map(go)} as UOp
  }
  
  return go(graph)

}


