import type { ScriptCtx } from "../main.ts";

export const main = async ({Tensor}: ScriptCtx) => {
  const N = 1000;
  const x = Tensor.rand([N,N])
  const y = Tensor.rand([N,N])
  const st = performance.now();
  const r = await x.matmul(y).sum().run()

  const GFLOPS = (performance.now() - st) * (N**3*2) / 1e9 * 1e3 

  return {
    GFLOPS
  }

}
