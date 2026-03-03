
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
  op: "REDUCE",
  bin: BinOp,
  srcs: [UOp],
  axis: number
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
  val: number
} | {
  op: "VIEW",
  srcs: [UOp],
  views: View[]
}
export type HighGraph = UOp & { op: "CONST" | "BUFFER" | BinOp | "REDUCE" | MoveOp }

export type LowGraph = UOp & { op: "CONST" | "BUFFER" | BinOp | "REDUCE" | "RANGE" | "ENDRANGE" | "INDEX" | "STORE" } & {srcs: LowGraph[]}



export type Op = UOp["op"]

export type Raw = number | Raw[]

export type Shape = number[]

export type ShapeTracker = {
  shape: Shape,
  strides: number[],
  mask: [number, number][],
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

export type Scheduler = {
  kernels: Kernel[]
  buffers: RAWBUFFER[]
  launch: () => Promise<RAWBUFFER>
}

export type Tensor = {
  realized?: RAWBUFFER
}
