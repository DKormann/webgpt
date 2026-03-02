import type { Tensor as TensorType, BACKEND as BackendType } from "../tensor.ts";

type Ctx = {
  Tensor: typeof TensorType;
  BACKEND: typeof BackendType;
  webgpuAvailable: boolean
  
};

export const main = async ({ Tensor, BACKEND, webgpuAvailable }: Ctx) => {
  BACKEND.default = webgpuAvailable ? "webgpu" : "js";
  const t = Tensor.new([[1, 2, 3], [4, 5, 6]]);
  return await t.run();
};
