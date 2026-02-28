export type MathOP = "ADD" | "MUL" | "IDX";
export type OP = MathOP | "RANGE" | "CONST" | "BUFFER";

export type UOP =
  | {
      op: "CONST";
      data: number[];
    }
  | {
      op: "BUFFER";
      idx: number;
    }
  | {
      op: "RANGE";
    }
  | {
      op: MathOP;
      srcs: [UOP, UOP];
    };
