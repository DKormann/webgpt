
import {describe, expect, test} from "bun:test"
import { WEBGPU } from "./webgpu"
import { LowGraph, UOp } from "./types"
import { uop } from "./uops"


describe("linearize ",()=>{

  let device= WEBGPU
  test("basic reduce",async ()=>{

    let buffer = device.createBuffer(10)

    let g: UOp = {
      op:"REDUCE",
      axis: 0,
      bin: "ADD",
      srcs: [
        {
          op: "VIEW",
          views: [
            {
              dims: [10],
              strides: [1],
            }
          ],
          srcs: [
            uop.buffer(buffer)
          ]
        }
      ]
    }

  })
})