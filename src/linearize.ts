import type { BUFFER, RAWBUFFER, UOp, UOpKind } from "./types";
import { uop } from "./uops";


type KernelUOp = UOp & { op: "KERNEL" };

export type Linearized = {
  kernels: KernelUOp[];
  buffers: RAWBUFFER[];
  output: RAWBUFFER;
};

export const linearize = (root: KernelUOp): UOpKind<"PROGRAM"> => {
  const topo = uop.topo(root);
  let kernels = topo.filter((x): x is KernelUOp => x.op === "KERNEL");
  if (kernels.length === 0) throw new Error("linearize expected at least one KERNEL");

  let getbuffer =(K:KernelUOp)=>{
    let st = K.srcs[0]

    if (st.op == "STORE"){
      let idx = st.srcs[1]
      if (idx.op == "INDEX") {
        let buf = idx.srcs[0]
        if (buf.op == "BUFFER") return buf as BUFFER 
      }
    }
    throw new Error("MALFORMD KERNEL" + uop.fmt(K))
  }

  kernels = kernels.map(x=>{
    let go = ((u:UOp):UOp=>{
      if (u.op == "KERNEL") return getbuffer(u)
      return {...u,srcs: u.srcs.map(go)} as UOp
    });
    return {...x,srcs: x.srcs.map(go)} as KernelUOp
  })

  return uop.dedup({
    op:"PROGRAM",
    srcs:kernels,
    out: getbuffer(root).buf
  }) as UOpKind<"PROGRAM">

};
