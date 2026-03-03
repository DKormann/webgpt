import { RAWBUFFER, UOp } from "./types";


export const uop
// : Record <string, (...args: any[]) => UOp>
= {

  add: (a:UOp, b:UOp):UOp=> ({op: "ADD", srcs:[a,b]}),

  buffer : (buf: RAWBUFFER):UOp & {op:"BUFFER"}=> ({
    op:"BUFFER",
    srcs:[],
    buf,
  }),


  range : (max:number):UOp & {op:"RANGE"} => ({op:"RANGE", srcs:[], max}),
  endrange : (range: UOp & {op: "RANGE"}) : UOp & {op:"ENDRANGE"} => ({op:"ENDRANGE", srcs:[range]}),

  const : (val: number): UOp & {op:"CONST"} => ({
    op: "CONST",
    val,
    srcs:[]
  }),


  store :  (src: UOp, dest: UOp, index:UOp) : UOp & {op:"STORE"} => ({
    op: "STORE",
    srcs: [
      src,
      dest,
      index
    ]
  }),
  index: (src: UOp, index: UOp): UOp => ({op:"INDEX", srcs:[src,index]})

}
