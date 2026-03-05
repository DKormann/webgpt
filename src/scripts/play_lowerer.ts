
import {Tensor } from "../tensor"
import {lowerer} from "../lowerer"
import { kernelize } from "../kernelize"
import { uop } from "../uops"
import { linearize } from "../linearize"
import { UOp } from "../types"
import { WEBGPU } from "../webgpu"
import { DEBUG } from "../debug"

DEBUG.set(1)


let log = (x:UOp)=>console.log(uop.fmt(x))
const buffersIn = (graph: UOp[]): (UOp & { op: "BUFFER" })[] => {
  const out: (UOp & { op: "BUFFER" })[] = [];
  const seen = new Set<UOp>();
  const walk = (u: UOp) => {
    if (u.op === "BUFFER" && !seen.has(u)) {
      seen.add(u);
      out.push(u as UOp & { op: "BUFFER" });
    }
    u.srcs.forEach(walk);
  };
  graph.forEach(walk);
  return out;
};

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

console.log("\nWGSL:")
li.forEach((s, i) => {
  const bufs = buffersIn(s.steps).map((b) => b.buf) as Parameters<typeof WEBGPU.createKernel>[1];
  console.log(`\n-- kernel ${i} --`);
  WEBGPU.createKernel(s.steps, bufs);
})




// console.log(uop.fmt(l))
