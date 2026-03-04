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

  test("matmul-shaped store lowers to nested output loops + inner reduce loop", async () => {
    const a = device.createBuffer(6)   // [2,3]
    const b = device.createBuffer(6)   // [3,2]
    const out = device.createBuffer(4) // [2,2]

    const aView = uop.view(uop.buffer(a), [{ dims: [2, 3, 2], strides: [3, 1, 0] }])
    const bView = uop.view(uop.buffer(b), [{ dims: [2, 3, 2], strides: [0, 2, 1] }])
    const outView = uop.view(uop.buffer(out), [{ dims: [2, 2], strides: [2, 1] }])

    const high = uop.store(
      uop.reduce(uop.mul(aView, bView), 1, "ADD"),
      outView
    )

    const low = linearize(high)

    const range0 = uop.range(2)
    const range1 = uop.range(2)
    const range2 = uop.range(3)

    const outIdx = uop.add(uop.mul(range0, uop.const(2)), range1)
    const aIdx = uop.add(uop.add(uop.mul(range0, uop.const(3)), range2), uop.mul(range1, uop.const(0)))
    const bIdx = uop.add(uop.add(uop.mul(range0, uop.const(0)), uop.mul(range2, uop.const(2))), range1)

    const expected: UOp[] = [
      range0,
      range1,
      uop.store(uop.const(0), uop.index(uop.buffer(out), outIdx)),
      range2,
      uop.store(
        uop.add(
          uop.index(uop.buffer(out), outIdx),
          uop.mul(
            uop.index(uop.buffer(a), aIdx),
            uop.index(uop.buffer(b), bIdx)
          )
        ),
        uop.index(uop.buffer(out), outIdx)
      ),
      uop.endrange(range2),
      uop.endrange(range1),
      uop.endrange(range0),
    ]

    expect(low).toEqual(expected)
  })
})
