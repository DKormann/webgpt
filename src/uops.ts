import { BinOp, RAWBUFFER, UOp, View } from "./types";


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

  rand : (size:number, seed: number):UOp => ({op:"RAND", seed, srcs:[], size}),

  view: (src: UOp, views: View[]): UOp & { op: "VIEW" } => ({
    op: "VIEW",
    srcs: [src],
    views
  }),

  reduce: (src: UOp, axis: number, bin: BinOp): UOp & { op: "REDUCE" } => ({
    op: "REDUCE",
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
  index: (buf: UOp, index: UOp): UOp => ({op:"INDEX", srcs:[buf,index]})
}
