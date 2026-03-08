import { DEBUG } from "../debug";
import { compile, Tensor, TensorVar } from "../tensor";

let N = 1000


let r = await Tensor.rand([N,N])


let matmul = compile((a)=>a.matmul(a))


const timeit = <T> (f:()=>Promise<T>) => {
  let st = performance.now()
  return f().then(res=> ({res, dt: performance.now()-st}))
}



DEBUG.set(1)


let res = [];
for (let i =0; i <10; i++){
  res.push(await timeit(()=>matmul(r)))
}


res.map((t,i) => {
  console.log(`run ${i}, time: ${t.dt/1e3}, GFLOPS: ${(N*N*N*2)/t.dt/1e9}`)
})


console.log((await res[9].res.read()).slice(0,10))