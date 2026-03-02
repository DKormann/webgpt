import { describe, expect, test } from "bun:test";
import { BACKEND, Tensor } from "./tensor.ts";
import { linear, matmul } from "./nn.ts";

BACKEND.default = "naive";

describe("nn helpers", () => {
  test("matmul helper parity with instance matmul", async () => {
    const a = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    const b = Tensor.new([[1, 2], [3, 4], [5, 6]]);
    expect(await matmul(a, b).run()).toEqual(await a.matmul(b).run());
  });

  test("Tensor.linear 2D + bias", async () => {
    const x = Tensor.new([[1, 2], [3, 4]]);
    const w = Tensor.new([[5, 6, 7], [8, 9, 10]]);
    const b = Tensor.new([1, 2, 3]);
    expect(await Tensor.linear(x, w, b).run()).toEqual([
      [22, 26, 30],
      [48, 56, 64]
    ]);
  });

  test("linear helper 1D + bias", async () => {
    const x = Tensor.new([1, 2]);
    const w = Tensor.new([[5, 6, 7], [8, 9, 10]]);
    const b = Tensor.new([1, 2, 3]);
    expect(await linear(x, w, b).run()).toEqual([22, 26, 30]);
  });
});
