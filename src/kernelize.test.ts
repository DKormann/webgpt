import { describe, expect, test } from "bun:test";
import { Tensor } from "./tensor";
import { kernelize } from "./kernelize";

describe("kernelize", () => {
  test("matmul tensor graph creates output and input buffers", async () => {

    
    const c = a.matmul(b);

    const out = kernelize(c);

    expect(out.graph.op).toBe("STORE");
    expect(out.buffers.length).toBe(3); // output + two inputs
    expect(out.output.size).toBe(4); // 2x2

    const inA = await out.buffers[1].read();
    const inB = await out.buffers[2].read();
    expect(inA).toEqual([1, 2, 3, 4, 5, 6]);
    expect(inB).toEqual([7, 8, 9, 10, 11, 12]);
  });
});
