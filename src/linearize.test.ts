import {describe, expect, test} from "bun:test"
import { WEBGPU } from "./webgpu"
import { UOp } from "./types"
import { uop } from "./uops"
import { linearize } from "./linearize"


describe("linearize",()=>{
  let device = WEBGPU

  test("basic reduce -> low graph ready for codegen",async ()=>{
    let input = device.createBuffer(10)
    let output = device.createBuffer(1)

    let ing:UOp = {
      op: "VIEW",
      views: [{ dims: [10], strides: [1] }],
      srcs: [uop.buffer(input)]
    }
    let outg:UOp = {
      op: "VIEW",
      views: [{ dims: [1], strides: [1] }],
      srcs: [uop.buffer(output)]
    }
    let store:UOp = uop.store(
      {
        op:"REDUCE",
        bin: "ADD",
        srcs: [ing],
        axis: 0
      },
      outg,
    )


    const low = linearize(store)
    let rang = uop.range(10)

    const kern:UOp[] = [
      rang,
      uop.store(
        uop.add(
          uop.index(
            uop.buffer(output),
            uop.const(0)
          ),
          uop.index(
            uop.buffer(input),
            rang
          )
        ),
        uop.buffer(output),
        uop.const(0)
      ),
      uop.endrange(rang)
    ] 

    expect(low).toEqual(kern)

  })
})
