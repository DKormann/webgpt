
import {Tensor } from "../tensor"
import {lowerer} from "../lowerer"
import { kernelize } from "../kernelize"
import { uop } from "../uops"
import { linearize } from "../linearize"
import { UOp } from "../types"
import { WEBGPU } from "../webgpu"
import { DEBUG } from "../debug"


let log = (x:UOp)=>console.log(uop.fmt(x))

let t = Tensor.rand([2,2])
let q = Tensor.rand([2,2])

let y = t.matmul(q).uop

console.log(uop.fmt(y))

let s = kernelize(y)

console.log(uop.fmt(s))


let l = lowerer(s)

log(l)

let li = linearize(l)

li.map(x=>console.log(x.toString()))




// console.log(uop.fmt(l))
