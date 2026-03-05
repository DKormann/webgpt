
import { DEBUG } from "../debug";
import { Tensor } from "../tensor";

DEBUG.set(1)

const N = 250

const a = Tensor.rand([N, N])
const b = Tensor.rand([N, N])


let st = performance.now()

let res = await a.matmul(b).sum().run("webgpu")

let dt = performance.now() - st

console.log({
  GFLOPS: (2 * N * N * N) / (dt / 1e3) / 1e9,
  seconds: dt / 1e3
})
