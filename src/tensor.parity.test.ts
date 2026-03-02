import { describe, test, expect } from "bun:test";
import type { Tensor } from "./tensor.ts";
import { Tensor as T } from "./tensor.ts";

const compare = async (t: Tensor) => {
  expect(await t.run("naive")).toEqual(await t.run("js"));
};



describe("tensor runtime parity", () => {
  test("shape chain parity", async () => {
    await compare(
      T.new([[1, 2, 3], [4, 5, 6]])
        .permute([1, 0])
        .pad([[1, 0], [0, 1]])
        .shrink([[0, 3], [0, 2]])
        .reshape([2, 3])
        .expand([2, 3])
        .add(T.const(1, [2, 3]))
        .sum([1])
    );
  });

  test("stride mismatch add parity", async () => {
    const a = T.new([[1, 2, 3], [4, 5, 6]]);
    await compare(a.add(a.reshape([3, 2]).permute([1, 0])));
  });

  test("mul parity", async () => {
    const a = T.new([[1, 2, 3], [4, 5, 6]]);
    await compare(a.mul(T.const(2, [2, 3])));
  });

  test("reduce dims parity", async () => {
    const t = T.new([[1, 2, 3], [4, 5, 6]]);
    await compare(t.sum([0]));
    await compare(t.sum([1]));
    await compare(t.prod([0]));
    await compare(t.prod([1]));
  });

  test("matmul parity", async () => {
    const a = T.new([[1, 2, 3], [4, 5, 6]]);
    const b = T.new([[1, 2], [3, 4], [5, 6]]);
    await compare(a.matmul(b));
  });
});
