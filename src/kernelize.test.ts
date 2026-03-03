import { describe, expect, test } from "bun:test";
import { Tensor } from "./tensor";
import { kernelize } from "./kernelize";

describe("kernelize", () => {


  test("matmul kernelize", ()=>{
    const A = Tensor.rand([4,4])
    const B = Tensor.rand([4,4])

    const u = A.matmul(B).uop

    let sched = kernelize([u])

    expect(sched.items.length).toEqual(1)
    expect(sched.items[0].Buffers.length).toEqual(1)
    expect(sched.items[0].Buffers[0].size).toEqual(16)

  })
});
