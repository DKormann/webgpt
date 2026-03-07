import { Backend, mkUop, Runner, type BinOp, type BufferRef, type RAWBUFFER, type UOp } from "./types";
import { uop } from "./uops";
import { kernelize } from "./kernelize";
import { linearize } from "./linearize";
import { lowerer } from "./lowerer";
import { WEBGPU } from "./webgpu";


export type Raw = number | Raw[];
export type RuntimeName = "js" | "webgpu";

type Tensor = RAWBUFFER & {shape: number[]}
type TensorFun = (...xs:Tensor[]) => Tensor

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

function compile  (fn: (...args:TensorVar[])=>TensorVar): Promise<TensorFun> {
 
  let ctx : {x_shapes:number[][], buffers: Map<BufferRef, RAWBUFFER>, runner: Runner} | null = null
  return async (...xs:Tensor[]) =>{

    let buffers: Map<BufferRef, RAWBUFFER> = new Map()
    if (xs.length!=fn.length) throw new Error(`expected ${fn.length} inputs, got: ${xs.length}`)

    if (ctx == null){
      let x_shapes = xs.map(x=>x.shape)
      let inbuffs = xs.map((x,i)=>(mkUop("BUFFER", [], {size: x.size, slot:i})))
      let invars = inbuffs.map((b,i)=>new TensorVar(b, xs[i].shape))
      let graph = fn(...invars).uop
      let kg = kernelize(graph)
      let lg = lowerer(kg, (size)=>{
        let b = mkUop("BUFFER", [], {size, slot: xs.length + buffers.size })
        buffers.set(b, WEBGPU.createBuffer(size))
        return b
      })
      let sched = linearize(lg)
      let runner = await WEBGPU.createRunner(sched)

      ctx = {x_shapes, buffers, runner}
    }
    ctx.runner.run((ref)=>{
      if (ref.arg.slot<xs.length){
        return xs[ref.arg.slot]
      }
      if (!ctx!.buffers.get(ref)) throw new Error(`BUFFER ${ref.arg.slot} not found`)
      return ctx!.buffers.get(ref)!
    })

    throw new Error("TODO")
  }
}

function main(){
  
}