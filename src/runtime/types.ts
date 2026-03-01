import type { Shape, UOP } from "../uops.ts";

export type RuntimeExec = (uop: UOP, shape: Shape) => number[];
export type RuntimeExecAsync = (uop: UOP, shape: Shape) => Promise<number[]>;
export type RuntimeName = "js" | "naive" | "webgpu";
