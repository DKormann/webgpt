
import { PatternMatcher, UPat } from "./patter_matcher";
import { mkUop, PermuteUOp, VDim, View, ViewUOp, type Kernel, type UOp, type UOpKind } from "./types";
import { prod, contiguos, sum } from "./helpers";
import { uop } from "./uops";


let maxval = (d:VDim) => (d.size-1) * d.stride
let insize = (a:View) => prod(a.map(x=>x.size))
let outsize = (a:View) => sum(a.map(maxval))+ 1

let wrapMerge = <T>(x:T[], f:(a:T, b:T)=>T[]):T[]=>{
  let res = [x[x.length-1]]
  for (let i=x.length-2; i>= 0; i--) res = [...f(x[i], res[0]), ...res.slice(1)]
  return res
}

export let mergeView = (av:View, bv:View) : View[]=>{
  bv = wrapMerge(bv, (a, b) => a.stride == b.stride*b.size ? [{stride:b.stride, size:a.size*b.size}] : [a,b] )
  if (outsize(av) > insize(bv)) throw new Error(`input view is too large:${outsize(av)} vs ${insize(bv)}`)
  let incont = contiguos(bv.map(x=>x.size))
  let cv:View = []
  for (let a of av){
    let stride = 0
    if (maxval(a)){
      let bm = bv.map((b,i)=> ({...b,rel:a.stride/incont[i].stride}))
      .filter(x=>Number.isInteger(x.rel) && x.size >= x.rel * a.size)[0]
      if (bm == undefined) return [av, bv]
      stride = bm.rel * bm.stride
    }
    cv.push({...a, stride})
  }
  return [cv]
}

export let compact = (views:View[]) => wrapMerge(views, mergeView)

export const mkKernel = (g:UOp):UOp => mkUop("KERNEL", [g], {size:prod(uop.shape(g))})

let pm = new PatternMatcher([
  [new UPat("r", "RESHAPE", [new UPat("x")]), ({r, x}) => {
    let shape = (r as UOpKind<"RESHAPE">).arg.shape
    if (prod(uop.shape(x)) != prod(shape)) throw new Error("RESHAPE numel mismatch")
    return mkUop("VIEW", [x], {views:[contiguos(shape)]})
  }],
  [new UPat("p", "PERMUTE", [new UPat("x")]), ({p, x}) => {
    let str = contiguos(uop.shape(x))
    return mkUop("VIEW",[x], {views:[(p as PermuteUOp).arg.shape.map(p=>str[p])]})
  }],
  [new UPat("e", "EXPAND", [new UPat("x")]), ({e, x}) => {
    let shp = contiguos(uop.shape(x))
    let dims = (e as UOpKind<"EXPAND">).arg.shape.slice()
    if (dims.length != shp.length) throw new Error("EXPAND rank mismatch")
    for (let i = 0; i < shp.length; i++) if (shp[i].size != 1 && dims[i] != shp[i].size) throw new Error("EXPAND on sized dim")
    return mkUop("VIEW", [x], {views:[contiguos(dims)]})
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

  return mkKernel(pm.match(u)) as UOpKind<"KERNEL">

}
