import { mkUop, type Kernel, type Programm, type UOp, type UOpKind } from "./types";
import { uop } from "./uops";

type KernelUOp = UOpKind<"KERNEL">;


export function schedule_fmt (sched:Programm){
  return sched.srcs.map((item)=>"SCHEDULEITEM\n"+ item.srcs.map((u,i)=>`${String(i).padStart(4)}: ${u.op.padEnd(10)}: ${u.srcs.map(x=>item.srcs.indexOf(x)).join(", ").padEnd(10)}:`+
  ` ${
    Object.entries(u).map(([k,v])=>(k!="op" && k !="srcs" ? `${k}:${item.srcs.includes(v) ? item.srcs.indexOf(v) : JSON.stringify(v)}`: ''))
    .filter(x=>x)
    .join(" ")
  }`).join("\n"))
  .join("\n")
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
      uops = uops.map(x=>x==a?b:x)
      uops.forEach(u=>u.srcs = u.srcs.map(x=>x==a?b:x))
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

      let defreg:UOp = uop.defReg(reducer.bin == "ADD" ? 0 : 1)
      let increg :UOp = {op:reducer.bin, srcs: [defreg, reducer.srcs[0]]}
      let accreg:UOp = uop.store(increg, defreg)
      let usereg:UOp = uop.noop(defreg)
      replace(reducer, usereg)

      const ranges = uops.filter((x): x is UOpKind<"RANGE"> => x.op == "RANGE");
      specials = ranges.filter(r=>reducer.keep.includes(r.id));
      const loops = ranges.filter(r=>!specials.includes(r));
      specials.forEach((r, i) => replace(r, specials[i]));

      let loopbody = new Set<UOp> ([...loops]);
      let loopafter = new Set<UOp> ([usereg]);

      uops.forEach(u=>{
        if (u.op == "RANGE") return
        if (u.srcs.some(x=>loopafter.has(x))) loopafter.add(u)
        else if (u.srcs.some(x=>loopbody.has(x))) loopbody.add(u)
      })

      uops = [
        ...specials,
        defreg,
        ...uops.filter(x=> x.op != "RANGE" && !loopbody.has(x) && !loopafter.has(x)),
        ...uops.filter(x=>loopbody.has(x)),
        increg, accreg,
        ...loops.reverse().map(l=>uop.endrange(l as UOpKind<"RANGE">)),
        ...loopafter
      ]
    }

    if (specials.length>3) throw new Error("not implemented")
    const threadsPerDim = [128, 64, 16][specials.length] ?? 128
    specials.forEach((range: UOpKind<"RANGE">, i) => {
      const thread = Math.max(1, Math.min(range.max, threadsPerDim));
      const block = Math.max(1, Math.ceil(range.max / thread));
      replace(range, {op: "SPECIAL", srcs:[], extent: range.max, axis: i as 0|1|2, block, thread})
    });

    return mkUop("LINEAR", uops)
  }))
};
