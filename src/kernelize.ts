
import { PatternMatcher, UPat } from "./patter_matcher";
import { mkUop, type Kernel, type UOp, type UOpKind } from "./types";
import { numel, stridesFor } from "./helpers";
import { uop } from "./uops";


export const mkKernel = (g:UOp):UOp => mkUop("KERNEL", [g], {size:numel(uop.shape(g))})

let pm = new PatternMatcher([
  [new UPat("r", "RESHAPE", [new UPat("x")]), ({r, x}) => {
    let olddims = uop.shape(x)
    let shape = (r as UOpKind<"RESHAPE">).arg.shape
    if (numel(olddims) != numel(shape)) throw new Error("RESHAPE numel mismatch")
    return mkUop("VIEW", [x], {views:[{dims: shape, strides: stridesFor(shape)}]})
  }],
  [new UPat("p", "PERMUTE", [new UPat("x")]), ({p, x}) => {
    let olddims = uop.shape(x)
    let dims = olddims.slice()
    let oldstrides = stridesFor(olddims)
    let strides = oldstrides.slice();
    ;(p as UOpKind<"PERMUTE">).arg.shape.map((pd: number, i: number)=>{
      dims[i] = olddims[pd]
      strides[i] = oldstrides[pd]
    })
    return mkUop("VIEW", [x], {views:[{dims, strides}]})
  }],
  [new UPat("e", "EXPAND", [new UPat("x")]), ({e, x}) => {
    let olddims = uop.shape(x)
    let oldstrides = stridesFor(olddims)
    let dims = (e as UOpKind<"EXPAND">).arg.shape.slice()
    let strides = dims.map((_x:number)=>0)
    let pad = dims.length - olddims.length
    if (pad < 0) throw new Error("EXPAND rank mismatch")
    for (let i = 0; i < olddims.length; i++) {
      let j = i + pad
      if (olddims[i] == dims[j]) strides[j] = oldstrides[i]
      else if (olddims[i] == 1) strides[j] = 0
      else throw new Error("EXPAND dim mismatch")
    }
    return mkUop("VIEW", [x], {views:[{dims, strides}]})
  }],
  [new UPat("v1", "VIEW", [new UPat("v2", "VIEW")]), ({v1,v2})=>(mkUop("VIEW", [v2.srcs[0]!], {views: ([v1,v2] as UOpKind<"VIEW">[]).map(v=>v.arg.views).flat()}))],
  [new UPat("x", "VIEW", [new UPat()]), ({x})=>{

    if (["KERNEL", "RAND", "BUFFER"].includes(x.srcs[0]!.op)) return null
    return {...x,srcs:[mkKernel(x.srcs[0]!)]} as UOp
  }],
])

export const kernelize = (u:UOp):UOpKind<"KERNEL"> => {

  return mkKernel(pm.match(u)) as UOpKind<"KERNEL">

}
