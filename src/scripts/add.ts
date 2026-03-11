import { DEBUG } from "../debug";
import { compile, Tensor, TensorVar } from "../tensor";
DEBUG.set(1)


let m = await compile(()=>TensorVar.rand([2,2]).add(TensorVar.rand([2,2])))()

console.log(await m.read())
