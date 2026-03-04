
import {Tensor } from "../tensor"

import {lowerer} from "../lowerer"
import { kernelize } from "../kernelize"
import { WEBGPU } from "../webgpu"
import { uop } from "../uops"



let t = Tensor.rand([2,2])
let q = Tensor.rand([2,2])

let y = t.matmul(q)

let s = kernelize(y, WEBGPU.createBuffer).items[0].roots[0]

console.log(uop.fmt(s))


let l = lowerer(s)

console.log(uop.fmt(l))

