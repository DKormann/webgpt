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

  const t = Tensor.new([[1,2,3],[4,5,6]])
  test("shape", ()=>{
    expect(t.shape).toEqual({
      dims: [2,3],
      strides: [3,1],
      numel:6
    })
  })

  test("reshape", () => {
    let r = t.reshape([3,2]);
    expect(r.run()).toEqual([1,2,3,4,5,6]);
    expect(r.shape).toEqual({
      dims:[3,2],
      strides:[2,1],
      numel: 6
    })
  });

  test("permute", () => {
    const r = t.permute([1,0]);
    expect(r.run()).toEqual([1,4,2,5,3,6]);
    expect(r.shape).toEqual({
      dims: [3,2],
      strides: [1,3],
      numel: 6
    })
  });

  test("expand", () => {
    const t = Tensor.new([[1,2,3]]).expand([2,3]);
    expect(t.run()).toEqual([1,2,3,1,2,3]);
  });

})
