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
      ...uop.view(
        uop.buffer(input),
        [{ dims: [10], strides: [1] }]
      )
    }
    let outg:UOp = uop.view(
      uop.buffer(output),
      [{ dims: [1], strides: [1] }]
    )
    let store:UOp = uop.store(
      uop.reduce(ing, 0, "ADD"),
      outg
    )

    const low = linearize(store)
    let rang = uop.range(10)

    const kern:UOp[] = [
      uop.store(
        uop.const(0),
        uop.index(uop.buffer(output), uop.const(0))
      ),
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
        uop.index(uop.buffer(output), uop.const(0))
      ),
      uop.endrange(rang)
    ] 

    expect(low).toEqual(kern)

  })
})
