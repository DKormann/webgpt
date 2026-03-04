
import { DEBUG } from "../debug";
import { Tensor } from "../tensor";

DEBUG.set(1)
// const a = Tensor.new([[1, 2, 3], [4, 5, 6]]);
const a = Tensor.rand([10,10])

await a.run("webgpu")
// const b = Tensor.new([[7, 8], [9, 10], [11, 12]]);
// const out = await a.matmul(b).run("webgpu");
// console.log(out)