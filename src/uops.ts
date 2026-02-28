export type Shape = {
  dims: number[];
  strides: number[];
  numel: number;
};

export type Binop = "ADD" | "MUL";

export type OP = Binop | "RANGE" | "CONST" | "IDX" | "REDUCE";

export type UOP =
  | {
      op: "CONST";
      data: number[];
    }
  | {
      op: "RANGE";
      count: number
    }
  | {
      op: "REDUCE";
      bin: Binop,
    }
  | {
      op: Binop;
      srcs: [UOP, UOP];
    };
