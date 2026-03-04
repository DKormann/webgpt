import { Tensor } from "./tensor";
import type { RAWBUFFER, Schedule, UOp } from "./types";
import { uop } from "./uops";


export const kernelize = (t:Tensor, alloc: (size:number)=>RAWBUFFER):Schedule => {

  let root = t.uop

  let sinkbuff = alloc(t.shape.dims.reduce((a,b)=>a*b,1))
  let store = uop.store(root, uop.buffer(sinkbuff))

  let buffs = new Set<RAWBUFFER>([sinkbuff])
  let getbuffs = (graph:UOp)=>{
    if (graph.op == "BUFFER") buffs.add(graph.buf)
    graph.srcs.forEach(getbuffs)
  }
  getbuffs(root)
  return {
    "items":[{
      Buffers:Array.from(buffs),
      roots: [store]
    }]
  }
}

