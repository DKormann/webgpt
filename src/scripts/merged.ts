import { compile, Tensor } from "../tensor";



let a = await Tensor.rand([2,2])

let f = compile(a=>{
  let e = a.mul(a)
  let f = a.matmul(a)

  return e.add(f).sum([1])
})





console.log(await (await f(a)).read())
