import { exec as jsExec } from "./js.ts";
import { exec as naiveExec } from "./naive.ts";
import type { RuntimeExec, RuntimeName } from "./types.ts";
import type { Shape, UOP } from "../uops.ts";

const runtimes: Record<RuntimeName, RuntimeExec> = {
  js: jsExec,
  naive: naiveExec
};

export type { RuntimeName };

export const exec = (runtime: RuntimeName, uop: UOP, shape: Shape): number[] =>
  runtimes[runtime](uop, shape);
