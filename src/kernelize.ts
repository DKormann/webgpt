
import { PatternMatcher, UPat } from "./patter_matcher";
import { mkUop, View, ViewUOp, type Kernel, type UOp, type UOpKind } from "./types";
import { numel, stridesFor } from "./helpers";
import { uop } from "./uops";


let insize = (a:View) => numel(a.dims)
let outsize = (a:View) => 1 + a.dims.map((d,i) => a.strides[i] * (d-1)).reduce((a,c)=>a+c)

let compose = (a:View, b:View) : View[]=>{
  if (outsize(a) > insize(b)) throw new Error("input view is too large")
  let instrides = stridesFor(b.dims);
  let strides = a.dims.map((d,ai)=>
    (a.strides[ai] == 0 || d == 1) ? 0 : b.strides.filter((bs,bi)=>b.dims[bi] >= d && instrides[bi] == a.strides[ai])[0])
  if (strides.some(x=>x==undefined)) return [a,b]
  return [{...a, strides}]
}

let compact = (views:View[]) =>{
  let nv : View[]= [views[views.length-1]]
  for (let i = views.length-2; i>=0; i--) nv = [...compose(views[i], nv[0] ), ...nv.slice(1)]
  return nv
}

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

  [new UPat("v", "VIEW"), ({v})=>{
    let vv = v as ViewUOp;
    let c = compact(vv.arg.views)
    return (c.length < vv.arg.views.length) ? {...vv, arg:{views:c}}:null
  }],

  [new UPat("x", "VIEW", [new UPat()]), ({x})=>{

    if (["KERNEL", "RAND", "BUFFER"].includes(x.srcs[0]!.op)) return null
    return {...x,srcs:[mkKernel(x.srcs[0]!)]} as UOp
  }],
])

export const kernelize = (u:UOp):UOpKind<"KERNEL"> => {

  console.log("KERNELIZE: ", uop.fmt(u))

  return mkKernel(pm.match(u)) as UOpKind<"KERNEL">

}
