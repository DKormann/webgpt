
import { DEBUG } from "../debug";
import { Tensor } from "../tensor";

DEBUG.set(1)

const N = 5

const a = Tensor.rand([N, N])
const b = Tensor.rand([N, N])


let st = performance.now()

let res = await a.matmul(b).sum().run("webgpu")

let dt = performance.now() - st

console.log({
  GFLOPS: dt / 1e3 / 1e9 * 2 * N * N * N,
  seconds: dt / 1e3
})

