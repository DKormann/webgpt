import { DEBUG } from "../debug";
import { compile, Tensor, TensorVar } from "../tensor";

DEBUG.set(1)

let N = 1000



let r = await Tensor.rand([N,N])


let matmul = compile((a)=>a.matmul(a))


await matmul(r)


let st = performance.now()


let RUNS = 10

for (let i =0; i <RUNS; i++){
  await matmul(r)
}

let t = {dt: (performance.now()-st) / RUNS}

console.log(`time: ${String(t.dt/1e3).slice(0,10)}, GFLOPS: ${String((N*N*N*2)/t.dt/1e9).slice(0,10)}`)
