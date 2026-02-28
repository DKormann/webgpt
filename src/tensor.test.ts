import { describe, test, expect } from "bun:test";
import {  Tensor } from "./tensor.ts";





describe("tensor math", () => {

  const A = Tensor.const(2, [2,2]);
  const B = Tensor.const(3, [2,2]);

  test("const", () => {
    expect(A.run()).toEqual([2,2,2,2]);
  });


  test("adding", () => {
    expect (A.add(B).run()).toEqual([5,5,5,5]);
  });

  test("new tensor", ()=>{
    expect(Tensor.new([1,2,3]).run()).toEqual([1,2,3])
  })



});


describe("tensor shape operations", ()=>{


  const A = Tensor.const(1, [2,3])

  

})

