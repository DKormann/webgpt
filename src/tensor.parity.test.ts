import { describe, test, expect } from "bun:test";
import type { Tensor } from "./tensor.ts";
import { Tensor as T } from "./tensor.ts";




const compare = (t:Tensor) => {
  expect(t.run("naive")).toEqual(t.run("js"))
}



describe("tensor runtime parity", () => {
  test("shape chain parity", () => {
    compare(
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

  test("stride mismatch add parity", () => {
    const a = T.new([[1, 2, 3], [4, 5, 6]]);
    compare(a.add(a.reshape([3, 2]).permute([1, 0])));
  });

  test("reduce dims parity", () => {
    const t = T.new([[1, 2, 3], [4, 5, 6]]);
    compare(t.sum([0]));
    compare(t.sum([1]));
    compare(t.prod([0]));
    compare(t.prod([1]));
  });
});
