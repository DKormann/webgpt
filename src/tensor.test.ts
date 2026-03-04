import { describe, expect, test } from "bun:test";
import { Tensor } from "./tensor";
import { uop } from "./uops";

describe("tensor minimal api", () => {
  test("Tensor.const initializes scalar-filled tensor metadata", () => {
    const t = Tensor.const(3, [2, 2]);
    expect(t.shape.dims).toEqual([2, 2]);
    expect(t.shape.numel).toBe(4);
    expect(t.uop.op).toBe("VIEW");
  });

  test("Tensor add graph", ()=>{
    const t = Tensor.const(2,[2,2])
    expect(t.add(t).uop).toEqual(uop.add(t.uop,t.uop))
  })

  test("Tensor mul graph", ()=>{
    const t = Tensor.const(2,[2,2])
    expect(t.mul(t).uop).toEqual(uop.mul(t.uop,t.uop))
  })

  test("new(raw) infers shape", () => {
    const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    expect(t.shape.dims).toEqual([2, 3]);
    expect(t.uop.op).toBe("VIEW");
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

  test("sum(dims?) creates REDUCE uop with expected shape", () => {
    const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    const s0 = t.sum([1]);
    expect(s0.uop.op).toBe("REDUCE_AXIS");
    expect(s0.shape.dims).toEqual([2]);

    const sall = t.sum();
    expect(sall.uop.op).toBe("REDUCE_AXIS");
    expect(sall.shape.dims).toEqual([]);
    expect(sall.shape.numel).toBe(1);
  });

  test("shape ops update shape metadata", () => {
    const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    expect(t.reshape([3, 2]).shape.dims).toEqual([3, 2]);
    expect(t.permute([1, 0]).shape.dims).toEqual([3, 2]);
    expect(Tensor.new([[1, 2, 3]]).expand([2, 3]).shape.dims).toEqual([2, 3]);
    expect(t.pad([[1, 1], [1, 0]]).shape.dims).toEqual([4, 4]);
    expect(t.pad([[1, 1], [1, 0]]).shrink([[1, 3], [1, 4]]).shape.dims).toEqual([2, 3]);
  });

})

describe("tensor: end to end",()=>{

  test("matmul runs end-to-end on webgpu", async () => {
    const a = Tensor.new([[1, 2, 3], [4, 5, 6]]);
    const b = Tensor.new([[7, 8], [9, 10], [11, 12]]);
    const out = await a.matmul(b).run("webgpu");
    expect(out).toEqual([58, 64, 139, 154]);
  });

  test("rand runs end-to-end on webgpu", async () => {
    const out = await Tensor.rand([2, 2]).run("webgpu");
    expect(Array.isArray(out)).toBeTrue();
    const vals = out as number[];
    expect(vals.length).toBe(4);
    for (const v of vals) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  test("rand matmul rand (10x10) runs end-to-end on webgpu", async () => {
    const out = await Tensor.rand([10, 10]).matmul(Tensor.rand([10, 10])).run("webgpu");
    expect(Array.isArray(out)).toBeTrue();
    const vals = out as number[];
    expect(vals.length).toBe(100);
    for (const v of vals) expect(Number.isFinite(v)).toBeTrue();
  });

  test("sum runs end-to-end on webgpu", async () => {
    const out = await Tensor.new([[1, 2, 3], [4, 5, 6]]).sum().run("webgpu");
    expect(out).toEqual([21]);
  });



  test("matmul sum", async () => {
    // const out = await Tensor.new([[1, 2, 3], [4, 5, 6]]).sum().run("webgpu");
    const A = Tensor.rand([4,4])
    const B = Tensor.rand([4,4])
    await A.matmul(B).sum().run("webgpu")
  });
});
