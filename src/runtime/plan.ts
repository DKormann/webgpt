import type { UOP } from "../uops.ts";

export type Plan = {
  id: (uop: UOP) => string;
  refCount: (uop: UOP) => number;
};

export const buildPlan = (root: UOP): Plan => {
  const ids = new Map<UOP, string>();
  const refs = new Map<UOP, number>();
  const seen = new Set<UOP>();

  const addRef = (u: UOP) => refs.set(u, (refs.get(u) ?? 0) + 1);

  const visit = (u: UOP) => {
    if (!ids.has(u)) ids.set(u, `n${ids.size}`);
    if (seen.has(u)) return;
    seen.add(u);
    if (u.op === "REDUCE") {
      addRef(u.src);
      visit(u.src);
      return;
    }
    if (u.op === "ADD" || u.op === "MUL") {
      addRef(u.srcs[0]);
      addRef(u.srcs[1]);
      visit(u.srcs[0]);
      visit(u.srcs[1]);
    }
  };

  visit(root);
  refs.set(root, refs.get(root) ?? 1);

  return {
    id: (uop) => ids.get(uop) ?? "nx",
    refCount: (uop) => refs.get(uop) ?? 0
  };
};
