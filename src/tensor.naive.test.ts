import { describe, test, expect } from "bun:test";
import { BACKEND, Tensor } from "./tensor.ts";

BACKEND.default = "naive"

describe("tensor naive truth tests", () => {
  test("const", async () => {
    expect(await Tensor.const(2, [2, 2]).run()).toEqual([[2, 2], [2, 2]]);
  });

  test("add", async () => {
    const a = Tensor.const(2, [2, 2]);
    const b = Tensor.const(3, [2, 2]);
    expect(await a.add(b).run()).toEqual([[5, 5], [5, 5]]);
  });

  test("mul", async () => {
    const a = Tensor.const(2, [2, 2]);
    const b = Tensor.const(3, [2, 2]);
    expect(await a.mul(b).run()).toEqual([[6, 6], [6, 6]]);
  });

  test("rand", async () => {
    const r = await Tensor.rand([2, 3]).run();
    const flat = (r as number[][]).flat();
    expect(flat.length).toBe(6);
    expect(flat.every((x) => x >= 0 && x < 1)).toBe(true);
    expect(new Set(flat).size > 1).toBe(true);
  });

  test("shape ops", async () => {
    const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    expect(await t.reshape([3, 2]).run()).toEqual([[1, 2], [3, 4], [5, 6]]);
    expect(await t.permute([1, 0]).run()).toEqual([[1, 4], [2, 5], [3, 6]]);
    expect(await Tensor.new([[1, 2, 3]]).expand([2, 3]).run()).toEqual([[1, 2, 3], [1, 2, 3]]);
  });

  test("pad and shrink", async () => {
    const p = Tensor.new([[1, 2, 3], [4, 5, 6]]).pad([[1, 1], [1, 0]]);
    expect(await p.run()).toEqual([
      [0, 0, 0, 0],
      [0, 1, 2, 3],
      [0, 4, 5, 6],
      [0, 0, 0, 0]
    ]);
    expect(await p.shrink([[1, 3], [1, 4]]).run()).toEqual([[1, 2, 3], [4, 5, 6]]);
  });

  test("reducers with dims", async () => {
    const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    expect(await t.sum().run()).toEqual(21);
    expect(await t.prod().run()).toEqual(720);
    expect(await t.sum([0]).run()).toEqual([5, 7, 9]);
    expect(await t.sum([1]).run()).toEqual([6, 15]);
    expect(await t.prod([0]).run()).toEqual([4, 10, 18]);
    expect(await t.prod([1]).run()).toEqual([6, 120]);
  });
});
