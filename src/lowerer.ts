import { stridesFor } from "./helpers";
import { mkBuffer, mkUop, type Kernel, type UOp, type UOpKind, type View } from "./types";
import { unFlattenIndex, uop } from "./uops";

type Range = UOpKind<"RANGE">
let nextRangeId = 1
const mkRange = (max:number):Range => mkUop("RANGE", [], {id: nextRangeId++, max})
const c = (x:number) => uop.const([x])





const indexView = (views: View[], buf: UOp, rngs:Range[]) : UOp =>{
  if (rngs.length == 0) rngs = [mkRange(1)]
  if (views.length == 0) throw new Error("NO VIEW")
  let indexes:UOp[] = rngs
  let res:UOp | undefined = undefined;
  views.forEach((view,i) =>{

    res = uop.add(...indexes.map((x,i)=>uop.mul(x,view.strides[i]))) 
    if (views[i+1]) indexes = unFlattenIndex(res, views[i+1].dims)
  })
  if (res == undefined) throw new Error("no index")

  return uop.index(buf, res!)
}

export const lowerer = (graph: Kernel): UOpKind<"KERNEL"> =>{
  console.log("LOWER",uop.fmt(graph))
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
    if (u.op == "CONST"){
      if (u.arg.val.length > 1){
        if (rngs == null) rngs = [mkRange(u.arg.val.length)]
        if (rngs.length > 1 || rngs[0].arg.max != u.arg.val.length) throw new Error("wrong ranges")
        return [uop.index(u, rngs[0]), rngs]
      }
      return [u, []]
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
    console.log((u))
    if (u.op == "KERNEL"){
      let {srcs:[c]} = u;
      let [data,rs] = rangify(c)
      let dims = rs.map(r=>r.arg.max)
      let st = mkUop("STORE", [data, indexView([{dims, "strides": stridesFor(dims)}], mkBuffer(u.arg.size), rs)], undefined)
      u = {...u, srcs:[st]}
    }
    if (u.op == "CONST" && u.arg.val.length>1){
      let buf = mkBuffer(u.arg.val.length)
      let ass = u.arg.val.map((c,i)=> uop.store(uop.index(buf, uop.const([i])), uop.const([c])))
      u = uop.after(buf, ...ass)
    }

    return {...u, srcs:u.srcs.map(go)} as UOp
  }
  
  return go(graph) as UOpKind<"KERNEL">

}
