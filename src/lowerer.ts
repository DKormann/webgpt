import { DEBUG } from "./debug";
import { numel, stridesFor, zip } from "./helpers";
import { PatternMatcher, UPat } from "./patter_matcher";
import { mkBuffer, mkUop, type Kernel, type UOp, type UOpKind, type View } from "./types";
import { unFlattenIndex, uop } from "./uops";

type Range = UOpKind<"RANGE">
let nextRangeId = 1
const mkRange = (max:number):Range => mkUop("RANGE", [], {id: nextRangeId++, max})
const c = (x:number) => uop.const([x])


let const_fold = new PatternMatcher([
  [new UPat("x", "MUL", [new UPat("a"), new UPat("b", "CONST")]), ({x, a, b})=>{
    if (b.op =="CONST" && b.arg.val[0] == 1 && b.arg.val.length == 1) return a
    return null
  }],
  [new UPat("x", "DIV", [new UPat("a"), new UPat("b", "CONST")]), ({x, a, b})=>{
    if (b.op =="CONST" && b.arg.val[0] == 1 && b.arg.val.length == 1) return a
    return null
  }],
  [new UPat("x", "IDIV", [new UPat("a"), new UPat("b", "CONST")]), ({x, a, b})=>{
    if (b.op =="CONST" && b.arg.val[0] == 1 && b.arg.val.length == 1) return a
    return null
  }],
  [new UPat("x", "MUL", [new UPat("a"), new UPat("b", "CONST")]), ({x, a, b})=>{
    if (b.op =="CONST" && b.arg.val[0] == 0 && b.arg.val.length == 1) return uop.const([0], b.arg.dtype)
    return null
  }],
  [new UPat("x", "ADD", [new UPat("a"), new UPat("b", "CONST")]), ({x, a, b})=>{
    if (b.op =="CONST" && b.arg.val[0] == 0 && b.arg.val.length == 1) return a
    return null
  }],

])

const indexView = (views: View[], rngs:Range[]) : UOp =>{
  

  if (rngs.length == 0) rngs = [mkRange(1)]
  if (views.length == 0) throw new Error("NO VIEW")
  let indexes:UOp[] = rngs
  let res:UOp | undefined = undefined;

  views.forEach((view,i) =>{
    if (indexes.length != view.strides.length) throw new Error("MISSMATCH STRIDUDE"+" "+indexes.length+" "+view.strides.length)
    let xs = indexes.map((x,i)=>uop.mul(x,view.strides[i]));
    res = uop.add(...xs) 
    if (views[i+1]) indexes = unFlattenIndex(res, views[i+1].dims)
  })
  if (res == undefined) throw new Error("no index")
  return res!
}

export const lowerer = (graph: Kernel): UOpKind<"KERNEL"> =>{
  let rangify = (u:UOp, rngs: Range[] | null = null) : [UOp, Range[]] =>{
    if (u.op == "REDUCE_AXIS"){
      let {arg:{axis, bin}, srcs: [ch]} = u;
      let [c,shp] = rangify(ch)
      let keep = shp.map((x,i)=>!axis.includes(i)?mkRange(1):x)
      return [mkUop("REDUCE", [c], {bin, keep:keep.map(k=>k.arg.id)}) , keep]
    }

    if (u.op == "VIEW"){
      let {arg:{views : [v0]}, srcs:[s]} = u;
      if (rngs == null){
        rngs = v0.dims.map(mkRange)
      }
      let index = indexView(u.arg.views, rngs);
      return [(s.op == "RAND") ? {...s, srcs: [index]} : uop.index(s, index), rngs]
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
      return  [u.op == "RAND" ? {...u, srcs: [rngs[0] ] } : uop.index(u, rngs[0]), rngs]
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
      let [data,rs] = rangify(c)
      let dims = rs.map(r=>r.arg.max)
      let st = mkUop("STORE", [data, uop.index( mkBuffer(u.arg.size), indexView([{dims, "strides": stridesFor(dims)}], rs))], undefined)
      u = {...u, srcs:[st]}
    }
    if (u.op == "CONST" && u.arg.val.length>1){
      let buf = mkBuffer(u.arg.val.length)
      let ass = u.arg.val.map((c,i)=> uop.store(uop.index(buf, uop.const([i])), uop.const([c])))
      u = uop.after(buf, ...ass)
    }

    return {...u, srcs:u.srcs.map(go)} as UOp
  }
  
  let kern = go(graph)

  return const_fold.match(kern) as Kernel


}
