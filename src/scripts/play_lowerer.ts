
import {Tensor } from "../tensor"
import {lowerer} from "../lowerer"
import { kernelize } from "../kernelize"
import { uop } from "../uops"



let t = Tensor.rand([2,2])
let q = Tensor.rand([2,2])

let y = t.matmul(q).uop

console.log(uop.fmt(y))

let s = kernelize(y)

console.log(uop.fmt(s))


let l = lowerer(s)

console.log(uop.fmt(l))

