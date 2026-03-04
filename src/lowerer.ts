import { PatternMatcher, UPat } from "./patter_matcher";
import type { Op, RAWBUFFER, Schedule, UOp, View } from "./types";
import { uop } from "./uops";


type UOpKind <OP> = UOp & {op: OP}


let pm = new PatternMatcher([
  [new UPat("v1", "VIEW", [new UPat("v2", "VIEW")]), ({v1,v2})=>(uop.view(v2.srcs[0]!, ([v1,v2] as UOpKind<"VIEW">[]).map(v=>v.views).flat()))],

])



export const lowerer = (graph: UOp): UOp =>{

  return pm.match(graph)

}


