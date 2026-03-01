import type { Shape, UOP } from "../uops.ts";

export type RuntimeExec = (uop: UOP, shape: Shape) => number[];
export type RuntimeName = "js" | "naive";
