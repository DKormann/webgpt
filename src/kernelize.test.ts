import { describe, expect, test } from "bun:test";
import { Tensor } from "./tensor";
import { kernelize } from "./kernelize";
import { WEBGPU } from "./webgpu";

describe("kernelize", () => {


  test("matmul kernelize", ()=>{
    const A = Tensor.rand([4,4])
    const B = Tensor.rand([4,4])

    const u = A.matmul(B)

    let sched = kernelize(u, WEBGPU.createBuffer)

    expect(sched.items.length).toEqual(1)
    expect(sched.items[0].Buffers.length).toEqual(1)
    expect(sched.items[0].Buffers[0].size).toEqual(16)

  })
});
