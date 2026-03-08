import { compile, TensorVar } from "../tensor";

let fn = compile(()=>TensorVar.rand([2,2]))


console.log(await fn())