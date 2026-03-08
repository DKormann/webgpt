import { mkBuffer, Runner, type BinOp, type BufferRef, type RAWBUFFER, type UOp } from "./types";
import { uop } from "./uops";
import { kernelize } from "./kernelize";
import { linearize } from "./linearize";
import { lowerer } from "./lowerer";
import { WEBGPU } from "./webgpu";
import { numel } from "./helpers";

export type Raw = number | Raw[];
export type RuntimeName = "js" | "webgpu";

type Tensor = RAWBUFFER & {shape: number[]}
type TensorFun = (...xs:Tensor[]) => Promise<Tensor>



export const Tensor = {
  rand : (shape: number[])=> compile(()=>TensorVar.rand(shape))()
  
}


export class TensorVar {

  constructor(public uop:UOp, public shape: number[]){}

  static rand = (shape:number[]) => new TensorVar(uop.rand(0,numel(shape)), shape)

  static const = (val: number[]) => new TensorVar(uop.const(...val), [val.length])
  
  static bin = (op:BinOp, a:TensorVar, b:TensorVar)=> new TensorVar(uop.bin(op)(a.uop,b.uop), a.shape)
  mul = (other:TensorVar) => TensorVar.bin("MUL", this, other)
  add = (other:TensorVar) => TensorVar.bin("ADD", this, other)
  sum = (dims?: number[]) => {
    dims = dims ?? this.shape.map((x,i)=>i)
    return new TensorVar(uop.reduce(this.uop, "ADD", dims), this.shape.filter((d,i)=>!dims.includes(i)))
  }

  permute = (dims:number[]) => new TensorVar({op:"PERMUTE", shape: dims, srcs:[this.uop], }, dims.map(d=>this.shape[d]))
  reshape = (shape: number[]) => new TensorVar({op: "RESHAPE", shape, srcs:[this.uop]}, shape)
  expand = (shape:number[]) => new TensorVar({op:"EXPAND", shape, srcs:[this.uop]}, shape)
  matmul = (other: TensorVar) => {
    let [K,V] = this.shape
    let [V_, W] = other.shape
    if (V!=V_) throw new Error("matmul: V!=V_")
    return new TensorVar(
      uop.reduce(
        uop.mul(
          uop.view(this.uop, [{ dims: [K, V, W], strides: [V, 1, 0] }]),
          uop.view(other.uop, [{ dims: [K, V, W], strides: [0, W, 1] }]),
        ),
        "ADD",
        [1]
      ),
      [K, W]
    )
  }
}


export type TensorRef = {
  buffer:BufferRef,
  shape:number[]
}

export function compile  (fn: (...args:TensorVar[])=>TensorVar): TensorFun {
  let ctx: {X:TensorRef[], temp: Map<number, RAWBUFFER>, runner: Runner, Y:TensorRef} | null = null
  return async (...xs:Tensor[]) =>{

    if (ctx == null){
      let inbuffs: BufferRef[] = xs.map(x=>mkBuffer(x.size))
      let X = xs.map((x, i)=>({shape:x.shape, buffer:inbuffs[i]}))

      let invars = inbuffs.map((b,i)=>new TensorVar(b, xs[i].shape))
      let graph = fn(...invars)

      let kg = kernelize(graph.uop)
      let lg = lowerer(kg)
      let buffers = new Map(
        uop.topo(lg)
          .filter((x): x is BufferRef => x.op =="BUFFER")
          .map(r=>[r.arg.slot, WEBGPU.createBuffer(r.arg.size)] as [number,RAWBUFFER])
      )
      let sched = linearize(lg)
      let runner = await WEBGPU.createRunner(sched)
      const last = sched.srcs[sched.srcs.length - 1];
      const st = [...last.srcs].reverse().find((x): x is UOp & { op: "STORE" } => x.op === "STORE");
      if (!st) throw new Error("output store not found in last linear kernel");
      const buffer = uop.topo(st.srcs[1]).find((x): x is BufferRef => x.op === "BUFFER");
      if (!buffer) throw new Error("output buffer not found in " + uop.fmt(st));
      ctx = {X, temp:buffers, runner, Y: {shape: graph.shape, buffer}}
    }

    if (xs.length!=fn.length) throw new Error(`expected ${fn.length} inputs, got: ${xs.length}`)
    if (!ctx) throw new Error("NO CTX")

    let out = WEBGPU.createBuffer(ctx.Y.buffer.arg.size)
    await ctx.runner((ref)=>{
      let inin = ctx!.X.findIndex(x=>x.buffer.arg.slot == ref.arg.slot)
      let res: RAWBUFFER;
      if (inin>=0) res = xs[inin]
      else if (ref.arg.slot == ctx!.Y.buffer.arg.slot) res = out
      else res = ctx!.temp.get(ref.arg.slot)!
      if (!res) throw new Error(`BUFFER ${ref.arg.slot} not found`)
      return res
    })
    return Object.assign(out, { shape: ctx.Y.shape })
  }
}

function main(){
  let fn = compile((a,b)=>{

    let [K,L] = a.shape
    let [M,N] = b.shape
    if (M!=L) throw new Error(`shapes dont match`)
    a = a.reshape([K,L,1]).expand([K,L,N])
    b = b.reshape([1,L,N]).expand([K,L,N])
    let res = a.mul(b).sum([1])
    return res
  })

  let rand = compile(()=>TensorVar.rand([2,2]))
  let t = rand()

}
