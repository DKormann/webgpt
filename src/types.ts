
export type BinOp = "ADD" | "MUL"
export type MoveOp = "RESHAPE" | "EXPAND" | "PERMUTE" | "PAD" | "SHRINK"


export type BUFFER = {
  op: "BUFFER"
  srcs: [],
  slot: number,
  size: number
}


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
} | BUFFER | {
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
  op: BinOp
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
  op: "RAND",
  srcs: [],
  seed: number,
  size?: number
} | {
  op: "VIEW",
  srcs: [UOp],
  views: View[]
} | {
  op: "DEFINE_REG"
  srcs:[]
  default: number
} | {
  op: "KERNEL",
  size:number,
  srcs: UOp[],
  buffers: number[],
} | {
  op: "PROGRAM",
  srcs: UOp[]
  out: RAWBUFFER
}

export type HighGraph = UOp & { op: "CONST" | "BUFFER" | BinOp | "REDUCE_AXIS" | MoveOp }

export type LowGraph = UOp & { op: "CONST" | "RAND" | "BUFFER" | BinOp | "REDUCE_AXIS" | "RANGE" | "ENDRANGE" | "INDEX" | "STORE" | "DEFINE_REG" } & {srcs: LowGraph[]}



export type Op = UOp["op"]

export type Raw = number | Raw[]

export type Shape = number[]

export type ShapeTracker = {
  shape: Shape,
  strides: number[],
  mask: [number, number][],
}

export type Schedule = {
  items: {
    Buffers: RAWBUFFER[]
    roots: (UOp & {op:"STORE"})[]
  }[]
}

export type RAWBUFFER = {size: number, read: ()=>Promise<number[]>}

export type BACKEND <B extends RAWBUFFER> = {
  max_blocks: [number, number, number]
  max_threads: [number, number, number]
  createBuffer : (size: number) => B
  createRunner : (graph: UOp[]) => Runner
}

export type Runner = {
  run: (buffers: RAWBUFFER[]) => Promise<void>
}


export type Tensor = {
  realized?: RAWBUFFER
}
