
import { compile, Tensor } from "../tensor";


let a = await Tensor.rand([2,2])
let b = await compile(a=>a.permute([1,0]))(a)


console.log(await a.read())
console.log(await b.read())


