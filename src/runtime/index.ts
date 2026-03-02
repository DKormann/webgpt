import { exec as jsExec } from "./js.ts";
import { exec as naiveExec } from "./naive.ts";
import { execAsync as webgpuExecAsync } from "./webgpu.ts";
import type { RuntimeExecAsync, RuntimeName } from "./types.ts";
import type { Shape, UOP } from "../uops.ts";

const asyncRuntimes: Record<RuntimeName, RuntimeExecAsync> = {
  js: async (uop, shape) => jsExec(uop, shape),
  naive: async (uop, shape) => naiveExec(uop, shape),
  webgpu: webgpuExecAsync
};

export type { RuntimeName };

export const execAsync = (runtime: RuntimeName, uop: UOP, shape: Shape): Promise<number[]> => asyncRuntimes[runtime](uop, shape);
