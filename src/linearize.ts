import { RAWBUFFER, UOp } from "./types";
import { uop } from "./uops";




export const linearize = (graph: UOp) : UOp[] =>{
  if (graph.op !== "REDUCE") throw new Error(`linearize unsupported root op: ${graph.op}`);
  if (graph.bin !== "ADD") throw new Error(`linearize only supports ADD reduce for now`);
  if (graph.srcs.length !== 1) throw new Error("REDUCE expects one source");

  const src = graph.srcs[0];
  if (src.op !== "VIEW") throw new Error(`linearize expects VIEW source, got ${src.op}`);
  if (src.views.length !== 1) throw new Error("linearize expects exactly one VIEW descriptor");
  if (src.srcs.length !== 1) throw new Error("VIEW expects one source");

  const view = src.views[0];
  if (view.dims.length !== 1 || view.strides.length !== 1 || view.strides[0] !== 1) {
    throw new Error("linearize currently supports only 1D contiguous views");
  }

  const base = src.srcs[0];
  if (base.op !== "BUFFER") throw new Error(`VIEW source must be BUFFER, got ${base.op}`);

  const out = buffs[0] ?? base.buf;
  const outBuf = uop.buffer(out);
  const inBuf = uop.buffer(base.buf);

  const zero = uop.const(0);
  const range = uop.range(view.dims[0]);

  return [
    uop.store(zero, outBuf, zero),
    range,
    uop.store(
      uop.add(uop.index(outBuf, zero), uop.index(inBuf, range)),
      outBuf,
      zero
    ),
    uop.endrange(range)
  ];

}
