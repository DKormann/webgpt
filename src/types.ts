
export type BinOp = "ADD" | "MUL" | "DIV" | "MOD" | "IDIV"
export type MoveOp = "RESHAPE" | "EXPAND" | "PERMUTE" | "PAD" | "SHRINK"


export type VDim = {
  size: number,
  stride: number,
}

export type View = VDim[]

export type DTYPE = "int32" | "float32"

export type TensorShape = {
  dims: number[];
  strides: number[];
  numel: number;
  offset?: number;
  mask?: [number, number][];
}

export type Tagged <Tag extends string, SRC extends UOp[], Arg> = {
  op: Tag,
  srcs: SRC
  arg: Arg
}




export const mkUop = <OP extends UOp["op"]>(
  op: OP,
  srcs: Extract<UOp, { op: OP }>["srcs"],
  arg?: Extract<UOp, { op: OP }>["arg"]
) => {
  // let safs = srcs.map((x) =>(typeof x == "number" ? mkUop("CONST", [], {val:[x], dtype}) :  x)) as Extract<UOp, { op: OP }>["srcs"];
  srcs.forEach(x=>{if (x==undefined) throw new Error("undefined src " + op)})
  return ({op, srcs:srcs, arg}) as Extract<UOp, { op: OP }>}

let slotcount = 0
export const mkBuffer = (size:number) => mkUop("BUFFER", [], {size, slot: slotcount ++ })



export type UOpKind <OP extends UOp["op"]> = UOp & {op: OP}
export type BufferRef = Tagged<"BUFFER", [], {size:number, slot:number}>
export type Linear = Tagged<"LINEAR", UOp[], undefined>
export type Programm = Tagged<"PROGRAMM", Linear[], undefined>
export type Kernel = Tagged<"KERNEL", [UOp], {size:number}>
export type Rand = Tagged<"RAND", [UOp] | [], {seed:number, size:number}>
export type StoreUOp = Tagged<"STORE", [UOp, UOp], undefined>
export type SpecialUOp = Tagged<"SPECIAL", [], {axis: 0 | 1 | 2, extent: number, block: number, thread: number}>
export type RangeUOp = Tagged<"RANGE", [], {id: number, max: number}>
export type EndRangeUOp = Tagged<"ENDRANGE", [RangeUOp], undefined>
export type NoopUOp = Tagged<"NOOP", [UOp], undefined>
export type IndexUOp = Tagged<"INDEX", [UOp, UOp], undefined>
export type ReduceAxisUOp = Tagged<"REDUCE_AXIS", [UOp], {bin: "ADD", axis: number[]}> // sums at axes but keeps the dim as size 1
export type ReduceUOp = Tagged<"REDUCE", [UOp], {bin: "ADD", keep: number[]}> // sums apart from keep but keeps the others as size 1 dim
export type AddUOp = Tagged<"ADD", [UOp, UOp], undefined>
export type MulUOp = Tagged<"MUL", [UOp, UOp], undefined>
export type DivUOp = Tagged<"DIV", [UOp, UOp], undefined>
export type IdivUOp = Tagged<"IDIV", [UOp, UOp], undefined>
export type ModUOp = Tagged<"MOD", [UOp, UOp], undefined>
export type ReshapeUOp = Tagged<"RESHAPE", [UOp], {shape: number[]}>
export type ExpandUOp = Tagged<"EXPAND", [UOp], {shape: number[]}>
export type PermuteUOp = Tagged<"PERMUTE", [UOp], {shape: number[]}>
export type PadUOp = Tagged<"PAD", [UOp], {args: [number,number][]}>
export type ShrinkUOp = Tagged<"SHRINK", [UOp], {args: [number,number][]}>
export type ConstUOp = Tagged<"CONST", [], {val:number[], dtype:DTYPE}>
export type ViewUOp = Tagged<"VIEW", [UOp], {views: View[]}>
export type DefineRegUOp = Tagged<"DEFINE_REG", [], {default: number}>
export type AfterUOp = Tagged<"AFTER", [UOp, ...UOp[]], undefined>

export type UOp = BufferRef | Linear | Programm | Kernel
 | Rand
 | StoreUOp | SpecialUOp | RangeUOp | EndRangeUOp | NoopUOp | IndexUOp | ReduceAxisUOp
 | ReduceUOp | AddUOp | MulUOp | DivUOp | ModUOp | IdivUOp
 | ReshapeUOp | ExpandUOp | PermuteUOp | PadUOp | ShrinkUOp | ConstUOp | ViewUOp | DefineRegUOp | AfterUOp


export type Op = UOp["op"]

export type Raw = number | Raw[]

export type Shape = number[]

export type ShapeTracker = {
  shape: Shape,
  strides: number[],
  mask: [number, number][],
}

export type Runner = (getbuffer: (r:BufferRef)=> RAWBUFFER) => Promise<void>

export type RAWBUFFER = {size: number, read: ()=>Promise<number[]>}

export type Backend <B extends RAWBUFFER> = {
  max_blocks: [number, number, number]
  max_threads: [number, number, number]
  createBuffer : (size: number) => B
  createRunner : (sched: Programm) => Promise<Runner>
}
export type Tensor = {
  realized?: RAWBUFFER
}
