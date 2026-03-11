import { compile, TensorVar } from "../tensor";



let fn = compile(()=>{
  let x = TensorVar.const([2])
  let gs = x.backward(x,[x])[0]
  return gs!
})

let G = await fn()

console.log(G)
