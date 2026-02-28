


type mathOP = "ADD" | "MUL" | "IDX" | "RANGE"
type OP = mathOP | "CONST"

type UOP = {
  srcs: UOP[]
  op: mathOP
} | {
  op:"CONST"
  data: number[]
}



