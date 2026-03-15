import { describe, expect, test } from "bun:test";
import { compile, Tensor } from "./tensor";

const close = (a: number, b: number, eps = 1e-4) => Math.abs(a - b) <= eps;
const flat = (x: unknown): number[] => (x as number[][]).flat();

describe("tensor e2e", () => {
  test("permute executes end to end", async () => {
    const x = await Tensor.rand([2, 3]);
    const f = compile((a) => a.permute([1, 0]));
    const y = await f(x);

    expect(y.shape).toEqual([3, 2]);

    const xv = await x.read() as number[][];
    const yv = await y.read() as number [][];

    expect(yv.flat().length).toBe(xv.flat().length);
    yv.forEach((a, i) => a.forEach((v, j)=> expect(v).toBeCloseTo(xv[j][i]) ))
  });

  test.skip("matmul executes end to end", async () => {
    const a = await Tensor.rand([2, 3]);
    const b = await Tensor.rand([3, 4]);
    const f = compile((x, y) => x.matmul(y));
    const y = await f(a, b);

    expect(y.shape).toEqual([2, 4]);

    const av = flat(await a.read());
    const bv = flat(await b.read());
    const yv = flat(await y.read());

    const exp: number[] = [];
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 4; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += av[i * 3 + k]! * bv[k * 4 + j]!;
        exp.push(s);
      }
    }

    expect(yv.length).toBe(exp.length);
    yv.forEach((v, i) => expect(close(v, exp[i]!)).toBeTrue());
  });

  test("add executes end to end", async () => {
    const x = await Tensor.rand([2, 3]);
    const f = compile((a) => a.add(a));
    const out = await f(x);

    expect(out.shape).toEqual([2, 3]);

    const xv = flat(await x.read());
    const ov = flat(await out.read());
    const exp = xv.map((v) => 2 * v);

    expect(ov.length).toBe(exp.length);
    ov.forEach((v, i) => expect(close(v, exp[i]!)).toBeTrue());
  });

  test.skip("permute + matmul + add executes end to end", async () => {
    const x = await Tensor.rand([2, 3]);
    const f = compile((a) => {
      const y = a.permute([1, 0]).matmul(a);
      return y.add(y);
    });
    const out = await f(x);

    expect(out.shape).toEqual([3, 3]);

    const xv = flat(await x.read());
    const ov = flat(await out.read());
    const exp: number[] = [];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 2; k++) s += xv[k * 3 + i]! * xv[k * 3 + j]!;
        exp.push(2 * s);
      }
    }

    expect(ov.length).toBe(exp.length);
    ov.forEach((v, i) => expect(close(v, exp[i]!)).toBeTrue());
  });
});
