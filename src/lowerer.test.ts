import { describe, expect, test } from "bun:test";
import { Tensor } from "./tensor";
import { kernelize } from "./kernelize";

import { uop } from "./uops";
import type { RAWBUFFER, UOp } from "./types";

const alloc = (size: number): RAWBUFFER => {
  const arr = new Array<number>(size).fill(0);
  return { size, read: async () => arr.slice() };
};

const hasShapeOp = (node: UOp): boolean => {
  if (node.op === "RESHAPE" || node.op === "EXPAND" || node.op === "PERMUTE" || node.op === "PAD" || node.op === "SHRINK") return true;
  return node.srcs.some(hasShapeOp);
};

describe("lower", () => {
  test("materializes VIEW(CONST) into buffer-backed view", () => {
    const t = Tensor.new([[1, 2], [3, 4]]);
    const sched = kernelize(t, alloc);
    const out = lower(sched);

    expect(out.items.length).toBe(1);
    expect(out.items[0]!.Buffers.length).toBe(2); // sink + const materialization
    const src = out.items[0]!.roots[0]!.srcs[0];
    expect(src.op).toBe("VIEW");
    if (src.op !== "VIEW") throw new Error("expected VIEW");
    expect(src.srcs[0]!.op).toBe("BUFFER");
  });

  test("lowers RESHAPE to VIEW in schedule roots", () => {
    const input = alloc(6);
    const output = alloc(6);
    const root = uop.store(
      { op: "RESHAPE", srcs: [uop.view(uop.buffer(input), [{ dims: [2, 3], strides: [3, 1] }])], shape: [3, 2] },
      uop.buffer(output)
    );
    const sched = { items: [{ Buffers: [output, input], roots: [root] }] };
    const out = lower(sched);

    expect(hasShapeOp(out.items[0]!.roots[0]!)).toBeFalse();
    expect(out.items[0]!.roots[0]!.srcs[0].op).toBe("VIEW");
  });
});
