import { exec } from "./runtime.ts";
import type { UOP } from "./uop.ts";

type TensorMethods = {
  add: (b:Tensor) => Tensor
  run: ()=>number[]
}

export type TensorData = {
  uop: UOP
  shape: number[]
  strides: number[]
  numel: number,
} 

export type Tensor = TensorData & TensorMethods

export const newTensor = (value: number, shape: number[]):Tensor =>mkTensor({
  uop: {op:"CONST", data: [value]},
  shape,
  strides: shape.map((_, i) => shape.slice(i + 1).reduce((a, c) => a * c, 1)),
  numel: shape.reduce((a,c)=>a*c, 1),
})


const mkTensor = (t: TensorData):Tensor=>{
  return {
    ...t,
    add: (b)=> mkTensor({
      ...t,
      uop:{op:"ADD", srcs: [t.uop,b.uop]}
    }),
    run: ()=>exec(t.uop, t.numel)

  }
}
