import { describe, test, expect } from "bun:test";
import { newTensor } from "./tensor.ts";





describe("tensor math", () => {

  const A = newTensor(2, [2,2]);
  const B = newTensor(3, [2,2]);

  test("const", () => {
    expect(A.run()).toEqual([2,2,2,2]);
  });


  test("adding", () => {
    expect (A.add(B).run()).toEqual([5,5,5,5]);
  });
});

