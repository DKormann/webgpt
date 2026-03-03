import { describe, expect, test } from "bun:test";
import { Tensor } from "./tensor";
import { uop } from "./uops";
import { WEBGPU } from "./webgpu";

describe("tensor minimal api", () => {


  test("graph creation", ()=>{
    const t = Tensor.const(1, [2,2])
    expect(t.uop).toEqual(
      uop.view(
        uop.const(1),
        [
          {
            dims:[2,2],
            strides:[0,0],
          }
        ]
      )
    )

    expect (t.add(t)).toEqual(uop.add(t.uop, t.uop))
  })

  test("new(raw) infers shape and runs", async () => {
    const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    expect(t.shape.dims).toEqual([2, 3]);
    expect(await t.run()).toEqual([[1, 2, 3], [4, 5, 6]]);
  });

  test("rand(shape) creates right numel and range", async () => {
    const t = Tensor.rand([2, 3]);
    const out = await t.run();
    const flat = (out as number[][]).flat();
    expect(flat.length).toBe(6);
    expect(flat.every((x) => x >= 0 && x < 1)).toBe(true);
  });

  test("mul(other) works elementwise", async () => {
    const a = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    const b = Tensor.new([[2, 3, 4], [5, 6, 7]]);
    expect(await a.mul(b).run()).toEqual([[2, 6, 12], [20, 30, 42]]);
  });

  test("shape ops preserve view semantics", async () => {
    const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);

    expect(await t.reshape([3, 2]).run()).toEqual([[1, 2], [3, 4], [5, 6]]);
    expect(await t.permute([1, 0]).run()).toEqual([[1, 4], [2, 5], [3, 6]]);
    expect(await Tensor.new([[1, 2, 3]]).expand([2, 3]).run()).toEqual([[1, 2, 3], [1, 2, 3]]);

    const p = Tensor.new([[1, 2, 3], [4, 5, 6]]).pad([[1, 1], [1, 0]]);
    expect(await p.run()).toEqual([
      [0, 0, 0, 0],
      [0, 1, 2, 3],
      [0, 4, 5, 6],
      [0, 0, 0, 0]
    ]);
    expect(await p.shrink([[1, 3], [1, 4]]).run()).toEqual([[1, 2, 3], [4, 5, 6]]);
  });
});
