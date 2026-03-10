
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



export const mkUop = <OP extends UOp["op"]>(
  op: OP,
  srcs: Extract<UOp, { op: OP }>["srcs"],
  arg: Extract<UOp, { op: OP }>["arg"]
) => ({op, srcs, arg}) as Extract<UOp, { op: OP }>

let slotcount = 0
export const mkBuffer = (size:number) => mkUop("BUFFER", [], {size, slot: slotcount ++ })

export type UOpKind <OP extends UOp["op"]> = UOp & {op: OP}

type Store = Tagged<"STORE", [UOp, UOp], undefined>
type Special = Tagged<"SPECIAL", [], {axis: 0 | 1 | 2, extent: number, block: number, thread: number}>
type Range = Tagged<"RANGE", [], {id: number, max: number}>
type EndRange = Tagged<"ENDRANGE", [Range], undefined>
type Noop = Tagged<"NOOP", [UOp], undefined>
type Index = Tagged<"INDEX", [UOp, UOp], undefined>
type ReduceAxis = Tagged<"REDUCE_AXIS", [UOp], {bin: "ADD", axis: number[]}> // sums at axes but keeps the dim as size 1
type Reduce = Tagged<"REDUCE", [UOp], {bin: "ADD", keep: number[]}> // sums apart from keep but keeps the others as size 1 dim
type Add = Tagged<"ADD", [UOp, UOp], undefined>
type Mul = Tagged<"MUL", [UOp, UOp], undefined>
type Div = Tagged<"DIV", [UOp, UOp], undefined>
type Mod = Tagged<"MOD", [UOp, UOp], undefined>
type Reshape = Tagged<"RESHAPE", [UOp], {shape: number[]}>
type Expand = Tagged<"EXPAND", [UOp], {shape: number[]}>
type Permute = Tagged<"PERMUTE", [UOp], {shape: number[]}>
type Pad = Tagged<"PAD", [UOp], {args: [number,number][]}>
type Shrink = Tagged<"SHRINK", [UOp], {args: [number,number][]}>
type Const = Tagged<"CONST", [], number[]>
type ViewUOp = Tagged<"VIEW", [UOp], {views: View[]}>
type DefineReg = Tagged<"DEFINE_REG", [], {default: number}>

export type UOp = BufferRef | Linear | Programm | Kernel
 | Rand
 | Store | Special | Range | EndRange | Noop | Index | ReduceAxis
 | Reduce | Add | Mul | Div | Mod
 | Reshape | Expand | Permute | Pad | Shrink | Const | ViewUOp | DefineReg


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
