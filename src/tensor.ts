import { Backend, mkBuffer, mkUop, Runner, type BinOp, type BufferRef, type RAWBUFFER, type UOp } from "./types";
import { uop } from "./uops";
import { kernelize } from "./kernelize";
import { linearize } from "./linearize";
import { lowerer } from "./lowerer";
import { WEBGPU } from "./webgpu";


export type Raw = number | Raw[];
export type RuntimeName = "js" | "webgpu";

type Tensor = RAWBUFFER & {shape: number[]}
type TensorFun = (...xs:Tensor[]) => Promise<Tensor>

class TensorVar {

  constructor(public uop:UOp, public shape: number[]){}
  
  static bin = (op:BinOp, a:TensorVar, b:TensorVar)=> new TensorVar(uop.bin(op)(a.uop,b.uop), a.shape)
  mul = (other:TensorVar) => TensorVar.bin("MUL", this, other)
  sum = (dims?: number[]) => {
    dims = dims ?? this.shape.map((x,i)=>i)
    new TensorVar(uop.reduce(this.uop, "ADD", dims), this.shape.filter((d,i)=>!dims.includes(i)))
  }

  permute = (dims:number[]) => new TensorVar({op:"PERMUTE", shape: dims, srcs:[this.uop], }, dims.map(d=>this.shape[d]))
  reshape = (shape: number[]) => new TensorVar({op: "RESHAPE", shape, srcs:[this.uop]}, shape)
  expand = (shape:number[]) => new TensorVar({op:"EXPAND", shape, srcs:[this.uop]}, shape)
  matmul = (other: TensorVar) => {
    let [K,V] = this.shape
    let [V_, W] = other.shape
    if (V!=V_) throw new Error("matmul: V!=V_")
    return this.reshape([K,V,1]).expand([K,V,W])
    .mul(this.reshape([1,V,W]).expand([K,V,W]))
    .sum([1])
  }
}


export type TensorRef = {
  buffer:BufferRef,
  shape:number[]
}

function compile  (fn: (...args:TensorVar[])=>TensorVar): TensorFun {
  let ctx: {X:TensorRef[], temp: Map<BufferRef, RAWBUFFER>, runner: Runner, Y:TensorRef} | null = null
  return async (...xs:Tensor[]) =>{

    if (ctx == null){
      let X = xs.map(x=>({shape:x.shape, buffer:mkBuffer(x.size)}))
      let inbuffs: BufferRef[] = xs.map(x=>(mkBuffer(x.size)))

      let invars = inbuffs.map((b,i)=>new TensorVar(b, xs[i].shape))
      let graph = fn(...invars)

      let kg = kernelize(graph.uop)
      let lg = lowerer(kg)
      let buffers = new Map(uop.topo(lg).filter(x=>x.op =="BUFFER").map(r=>[r, WEBGPU.createBuffer(r.arg.size)] as [BufferRef,RAWBUFFER]))
      let sched = linearize(lg)
      let runner = await WEBGPU.createRunner(sched)

      if (lg.srcs[0].op == "STORE"){
        let buffer = uop.topo(lg.srcs[0].srcs[1]).filter(x=>x.op == "BUFFER")[0]!
        if (!buffer) throw new Error("output buffer not found in "+uop.fmt(lg.srcs[0]))
        ctx = {X, temp:buffers, runner, Y: {shape: graph.shape, buffer}}
        
      }else throw("output buffer not found in "+uop.fmt(lg.srcs[0]))
    }

    if (xs.length!=fn.length) throw new Error(`expected ${fn.length} inputs, got: ${xs.length}`)
    if (!ctx) throw new Error("NO CTX")

    let out = WEBGPU.createBuffer(ctx.Y.buffer.arg.size)
    await ctx.runner((ref)=>{
      let inin = ctx!.X.findIndex(x=>x.buffer == ref)
      if (inin>=0) return xs[inin]
      if (ref == ctx!.Y.buffer) return out
      let res = ctx!.temp.get(ref)
      if (!res) throw new Error(`BUFFER ${ref.arg.slot} not found`)
      return res
    })
    return {...out, shape:ctx.Y.shape,}
  }
}

function main(){
  
}