import { describe, test, expect } from "bun:test";
import { Tensor } from "./tensor.ts";

describe("tensor autograd", () => {
  test("add backward", async () => {
    const a = Tensor.new([2, 3], { requiresGrad: true });
    const b = Tensor.new([5, 7], { requiresGrad: true });
    const y = a.add(b).sum();
    y.backward();
    expect(await a.grad!.run()).toEqual([1, 1]);
    expect(await b.grad!.run()).toEqual([1, 1]);
  });

  test("mul backward", async () => {
    const x = Tensor.new([2, 3], { requiresGrad: true });
    const y = x.mul(x).sum();
    y.backward();
    expect(await x.grad!.run()).toEqual([4, 6]);
  });

  test("sum dims backward", async () => {
    const x = Tensor.new([[1, 2, 3], [4, 5, 6]], { requiresGrad: true });
    const y = x.sum([1]).sum();
    y.backward();
    expect(await x.grad!.run()).toEqual([[1, 1, 1], [1, 1, 1]]);
  });
});
