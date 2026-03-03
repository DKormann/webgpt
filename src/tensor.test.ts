import { describe, expect, test } from "bun:test";
import { Tensor } from "./tensor";

describe("tensor minimal api", () => {
  test("Tensor.const initializes scalar-filled tensor metadata", () => {
    const t = Tensor.const(3, [2, 2]);
    expect(t.shape.dims).toEqual([2, 2]);
    expect(t.shape.numel).toBe(4);
    expect(t.uop.op).toBe("CONST");
  });

  test("Tensor add graph", ()=>{
    const t = Tensor.const(2,[2,2])
    expect(t.add(t))
  })

  test("new(raw) infers shape", () => {
    const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    expect(t.shape.dims).toEqual([2, 3]);
    expect(t.uop.op).toBe("CONST");
  });

  test("rand(shape) creates tensor with correct metadata", () => {
    const t = Tensor.rand([2, 3]);
    expect(t.shape.dims).toEqual([2, 3]);
    expect(t.shape.numel).toBe(6);
    expect(t.uop.op).toBe("RAND");
  });

  test("mul(other) creates MUL uop", () => {
    const a = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    const b = Tensor.new([[2, 3, 4], [5, 6, 7]]);
    const c = a.mul(b);
    expect(c.uop.op).toBe("MUL");
    expect(c.shape.dims).toEqual([2, 3]);
  });

  test("shape ops update shape metadata", () => {
    const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    expect(t.reshape([3, 2]).shape.dims).toEqual([3, 2]);
    expect(t.permute([1, 0]).shape.dims).toEqual([3, 2]);
    expect(Tensor.new([[1, 2, 3]]).expand([2, 3]).shape.dims).toEqual([2, 3]);
    expect(t.pad([[1, 1], [1, 0]]).shape.dims).toEqual([4, 4]);
    expect(t.pad([[1, 1], [1, 0]]).shrink([[1, 3], [1, 4]]).shape.dims).toEqual([2, 3]);
  });
});
