import { exec as jsExec } from "./js.ts";
import { exec as naiveExec } from "./naive.ts";
import { execAsync as webgpuExecAsync } from "./webgpu.ts";
import type { RuntimeExec, RuntimeExecAsync, RuntimeName } from "./types.ts";
import type { Shape, UOP } from "../uops.ts";

const runtimes: Record<Exclude<RuntimeName, "webgpu">, RuntimeExec> = {
  js: jsExec,
  naive: naiveExec
};
const asyncRuntimes: Record<RuntimeName, RuntimeExecAsync> = {
  js: async (uop, shape) => jsExec(uop, shape),
  naive: async (uop, shape) => naiveExec(uop, shape),
  webgpu: webgpuExecAsync
};

export type { RuntimeName };

export const exec = (runtime: RuntimeName, uop: UOP, shape: Shape): number[] => {
  if (runtime === "webgpu") throw new Error("webgpu runtime is async; use runAsync/execAsync");
  return runtimes[runtime](uop, shape);
};

export const execAsync = (runtime: RuntimeName, uop: UOP, shape: Shape): Promise<number[]> =>
  asyncRuntimes[runtime](uop, shape);
