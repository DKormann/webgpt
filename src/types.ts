
export type BinOp = "ADD" | "MUL"
export type MoveOp = "RESHAPE" | "EXPAND" | "PERMUTE" | "PAD" | "SHRINK"


export type BUFFER = {
  op: "BUFFER"
  srcs: [],
  buf: RAWBUFFER
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

export type UOpKind <OP> = UOp & {op: OP}


export type UOp = {
  op: "STORE",
  srcs: [
    UOp, // source
    UOp // destination
  ]
} | BUFFER | {
  op: "RANGE"
  srcs: [],
  max: number
} | {
  op: "ENDRANGE",
  srcs: [UOp & {op: "RANGE"}]
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
  keep: UOpKind<"RANGE">[]
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
} | {
  op: "KERNEL",
  size:number,
  srcs: UOp[],
  buffers: (UOp & { op: "BUFFER" })[],
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
  createBuffer : (size: number) => B
  createKernel : (graph: UOp[], buffers: B[]) => Kernel
}

export type Kernel = {
  graph: UOp[]
  buffers: RAWBUFFER[]
  launch: () => Promise<void>
}


export type Tensor = {
  realized?: RAWBUFFER
}
