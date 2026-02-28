export type Shape = {
  dims: number[];
  strides: number[];
  numel: number;
};

export type MathOP = "ADD" | "MUL";
export type OP = MathOP | "RANGE" | "CONST" | "IDX";

export type UOP =
  | {
      op: "CONST";
      data: number[];
    }
  | {
      op: "RANGE";
    }
  | {
      op: MathOP;
      srcs: [UOP, UOP];
    };
