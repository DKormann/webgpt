import { mkUop, type Kernel, type Programm, type UOp, type UOpKind } from "./types";
import { uop } from "./uops";

type KernelUOp = UOpKind<"KERNEL">;


export function schedule_fmt (sched:Programm){
  return sched.srcs.map((item)=>"SCHEDULEITEM\n"+ item.srcs.map((u,i)=>`${String(i).padStart(4)}: ${u.op.padEnd(10)}: ${u.srcs.map(x=>item.srcs.indexOf(x)).join(", ").padEnd(10)}:`+
  (u.arg?Object.entries(u.arg).map(([k,v])=>`${k}:${v}`).join(" "):"")

).join("\n")).join("\n")
}


export const linearize = (root: KernelUOp): Programm=> {
  const topo = uop.topo(root);
  let kernels : Kernel[] = topo.filter((x): x is KernelUOp => x.op === "KERNEL");
  if (kernels.length === 0) throw new Error("linearize expected at least one KERNEL");

  const getbuffer = (k: KernelUOp) => {
    const st = k.srcs[0];
    if (st.op === "STORE") {
      const idx = st.srcs[1];
      if (idx.op === "INDEX" && idx.srcs[0].op === "BUFFER") return idx.srcs[0];
    }
    throw new Error("MALFORMD KERNEL " + uop.fmt(k));
  };

  kernels = kernels.map((k) => {
    const replace = (u: UOp): UOp => {
      if (u.op === "KERNEL") return getbuffer(u);
      return { ...u, srcs: u.srcs.map(replace) } as UOp;
    };

    const st = replace(k.srcs[0]);
    if (st.op !== "STORE") throw new Error("kernel body must be STORE after lowering");

    return {...k, srcs:[st]}
  });

  return mkUop("PROGRAMM", kernels.map((kernel:UOp)=>{

    kernel = uop.dedup(kernel)
    let uops = uop.topo(kernel);
    const replace = (a:UOp, b:UOp) => {
      kernel = uop.map(kernel, x=> x.op == a.op && (JSON.stringify(x) == JSON.stringify(a)) ? b : x)
      kernel = uop.dedup(kernel)
      uops = uop.topo(kernel)
    }

    let reducer;
    let store: UOp;
    {
      let stores = uops.filter(x=>x.op == "STORE");
      if (stores.length != 1) throw new Error("expected exactly one STORE in kernel");
      store = stores[0];
      let reducers = uops.filter(x=>x.op == "REDUCE")
      if (reducers.length > 1) throw new Error("expected only one REDUCE in kernel");
      reducer = reducers[0]
    }

    let specials : UOpKind< "RANGE">[] = []

    if (!reducer){
      specials = uops.filter((x): x is UOpKind<"RANGE"> => x.op == "RANGE");
      uops = [...specials, ...uops.filter((x) => !specials.includes(x as UOpKind<"RANGE">))];
    }else{

      let  ranges = uops.filter((x)=> x.op == "RANGE")
      specials = ranges.filter(x=>reducer.arg.keep.includes(x.arg.id))
      let loops = ranges.filter(x=>!specials.includes(x))
      
      let defreg = uop.after(mkUop("DEFINE_REG", [], {default:reducer.arg.bin == "ADD" ? 0 : 1}), ...specials)
      let accreg = uop.after(mkUop("STORE", [mkUop(reducer.arg.bin, [defreg, reducer.srcs[0]]), defreg]), ...loops)
      
      let closeloops = loops.map(l=>mkUop("ENDRANGE", [l]))
      let usereg = uop.after(accreg, ...closeloops)
      replace(reducer, usereg)
      uops.forEach(x=>{if (x.op == "REDUCE") {throw new Error("REDUCE STILL IN UOPS")}})

      ranges = uops.filter((x)=> x.op == "RANGE")
      specials = ranges.filter(x=>reducer.arg.keep.includes(x.arg.id))
      loops = ranges.filter(x=>!specials.includes(x))

    }

    if (specials.length>3) throw new Error("not implemented")
    const threadsPerDim = [128, 64, 16][specials.length] ?? 128
    specials.forEach((range: UOpKind<"RANGE">, i) => {
      const thread = Math.max(1, Math.min(range.arg.max, threadsPerDim));
      const block = Math.max(1, Math.ceil(range.arg.max / thread));
      replace(range, mkUop("SPECIAL", [], {extent: range.arg.max, axis: i as 0|1|2, block, thread}))
    });

    return mkUop("LINEAR", uops, undefined)
  }), undefined)
};
