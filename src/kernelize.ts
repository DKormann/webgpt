import { PatternMatcher, UPat } from "./patter_matcher";
import type { UOp, UOpKind } from "./types";
import { uop } from "./uops";



export const findSize = (g:UOp):number=>{
  if ("size" in g) return g.size as number
  if (g.op == "CONST") return g.val.length
  if (g.op == "BUFFER") return g.size
  if (g.srcs.length == 0) throw new Error("cannot find size" + uop.fmt(g))
  return findSize(g.srcs[0] as UOp)
}


export const mkKernel = (g:UOp):UOp =>( {op:"KERNEL", size:findSize(g), srcs: [g], buffers: []})


let pm = new PatternMatcher([
  [new UPat("v1", "VIEW", [new UPat("v2", "VIEW")]), ({v1,v2})=>(uop.view(v2.srcs[0]!, ([v1,v2] as UOpKind<"VIEW">[]).map(v=>v.views).flat()))],
  [new UPat("x", "VIEW", [new UPat()]), ({x})=>{
    if (x.srcs[0]?.op == "KERNEL" || x.srcs[0]?.op == "RAND") return null
    return {...x,srcs:[mkKernel(x.srcs[0]!)]} as UOp
  }],
])

export const kernelize = (u:UOp):UOpKind<"KERNEL"> => {

  return mkKernel(pm.match(u)) as UOpKind<"KERNEL">

}
