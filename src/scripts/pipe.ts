import { asShape } from "../helpers";
import { compile, Tensor } from "../tensor";


let x = await Tensor.rand([2,3])

let f = compile(a=>{
  let y = a.permute([1,0]).matmul(a)
  return y.add(y)
})

let res = await f(x)

console.log(res.shape)
console.log(await res.read())

