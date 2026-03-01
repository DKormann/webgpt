import { describe, test, expect } from "bun:test";
import { BACKEND, Tensor } from "./tensor.ts";

BACKEND.default = "naive"

describe("tensor naive truth tests", () => {
  test("const", () => {
    expect(Tensor.const(2, [2, 2]).run()).toEqual([2, 2, 2, 2]);
  });

  test("add", () => {
    const a = Tensor.const(2, [2, 2]);
    const b = Tensor.const(3, [2, 2]);
    expect(a.add(b).run()).toEqual([5, 5, 5, 5]);
  });

  test("shape ops", () => {
    const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    expect(t.reshape([3, 2]).run()).toEqual([1, 2, 3, 4, 5, 6]);
    expect(t.permute([1, 0]).run()).toEqual([1, 4, 2, 5, 3, 6]);
    expect(Tensor.new([[1, 2, 3]]).expand([2, 3]).run()).toEqual([1, 2, 3, 1, 2, 3]);
  });

  test("pad and shrink", () => {
    const p = Tensor.new([[1, 2, 3], [4, 5, 6]]).pad([[1, 1], [1, 0]]);
    expect(p.run()).toEqual([0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 0, 0, 0]);
    expect(p.shrink([[1, 3], [1, 4]]).run()).toEqual([1, 2, 3, 4, 5, 6]);
  });

  test("reducers with dims", () => {
    const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    expect(t.sum().run()).toEqual([21]);
    expect(t.prod().run()).toEqual([720]);
    expect(t.sum([0]).run()).toEqual([5, 7, 9]);
    expect(t.sum([1]).run()).toEqual([6, 15]);
    expect(t.prod([0]).run()).toEqual([4, 10, 18]);
    expect(t.prod([1]).run()).toEqual([6, 120]);
  });
});
