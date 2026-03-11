
import { DEBUG } from "../debug";
import { compile, Tensor, TensorVar } from "../tensor";

let N = 1000

DEBUG.set(1)

let r = await Tensor.rand([N,N])


let matmul = compile((a)=>a.matmul(a))


await matmul(r)


let st = performance.now()


let RUNS = 10

for (let i =0; i <RUNS; i++){
  await matmul(r)
}

let avg_s = (performance.now()-st)/RUNS / 1e3

console.log(`time: ${String(avg_s).slice(0,10)}, GFLOPS: ${String((N*N*N*2)/avg_s/1e9).slice(0,10)}`)
