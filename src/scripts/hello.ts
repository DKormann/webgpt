import type { ScriptCtx } from "../main.ts";

export const main = async ({ Tensor, BACKEND, webgpuAvailable }: ScriptCtx) => {
  BACKEND.default = webgpuAvailable ? "webgpu" : "js";
  const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
  return await t.run();
};
