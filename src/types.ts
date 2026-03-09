
export type BinOp = "ADD" | "MUL"
export type MoveOp = "RESHAPE" | "EXPAND" | "PERMUTE" | "PAD" | "SHRINK"

export type View = {
  dims: number[],
  strides: number[]
}

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

export type BufferRef = Tagged<"BUFFER", [], {size:number, slot:number}>
export type Linear = Tagged<"LINEAR", UOp[], undefined>
export type Programm = Tagged<"PROGRAMM", Linear[], undefined>
export type Kernel = Tagged<"KERNEL", [UOp], {size:number}>
export type Rand = Tagged<"RAND", [UOp] | [], {seed:number, size:number}>


type highUOP = BufferRef | Linear | Programm | Kernel | Rand

export const mkUop = <OP extends UOp["op"]>(
  op: OP,
  srcs: Extract<UOp, { op: OP }>["srcs"],
  rest?: Omit<Extract<UOp, { op: OP }>, "op" | "srcs">
) => ({op, srcs, ...(rest ?? {})}) as Extract<UOp, { op: OP }>

let slotcount = 0
export const mkBuffer = (size:number) => mkUop("BUFFER", [], {arg: {size, slot: slotcount ++ }})

export type UOpKind <OP extends UOp["op"]> = UOp & {op: OP}


export type UOp = {
  op: "STORE",
  srcs: [
    UOp, // source
    UOp // destination
  ]
} | {
  op: "SPECIAL"
  srcs: [],
  axis: 0 | 1 | 2,
  extent: number,
  block: number,
  thread: number,
} |highUOP| {
  op: "RANGE"
  srcs: [],
  id: number,
  max: number
} | {
  op: "ENDRANGE",
  srcs: [UOp & {op: "RANGE"}]
} | {
  op: "NOOP",
  srcs: [UOp]
} | {
  op: "INDEX",
  srcs: [UOp, UOp]
} | {
  op: "REDUCE_AXIS",
  bin: BinOp,
  srcs: [UOp],
  axis: number[]
} | {
  op: "REDUCE",
  bin: BinOp,
  srcs: [UOp],
  keep: number[]
} | {
  op: "ADD"
  srcs: [UOp, UOp]
} | {
  op: "MUL"
  srcs: [UOp, UOp]
} | {
  op: "RESHAPE" | "EXPAND" | "PERMUTE"
  srcs: [UOp]
  shape: number[]
} | {
  op: "PAD" | "SHRINK"
  srcs: [UOp]
  args: [number,number] []
} | {
  op: "CONST",
  srcs: [],
  val: number[],
} | {
  op: "VIEW",
  srcs: [UOp],
  views: View[]
} | {
  op: "DEFINE_REG"
  srcs:[]
  default: number
}



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
