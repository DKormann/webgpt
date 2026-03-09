import type { UOp } from "./types";

export const uop
= {
  fmt: (u:UOp) : string => {
    const counts = new Map<UOp, number>()
    const walk = (x: UOp) => {
      counts.set(x, (counts.get(x) ?? 0) + 1)
      x.srcs.forEach(walk)
    }
    walk(u)

    const shared = new Set<UOp>(Array.from(counts.entries()).filter(([, c]) => c > 1).map(([x]) => x))
    const names = new Map<UOp, string>()
    let next = 0
    shared.forEach((x) => names.set(x, `v${next++}`))

    const emitted = new Set<UOp>()

    const head = (x: UOp): string => {
      let h = x.op
      Object.entries(x).forEach(([k, v]) => {
        if (!["srcs", "op", "seed"].includes(k)) h += ` ${k}:${JSON.stringify(v)}`
      })
      return h
    }

    const render = (x: UOp, indent = ""): string => {
      if (shared.has(x)) {
        const n = names.get(x)!
        if (emitted.has(x)) return `${indent}${n}`
        emitted.add(x)
        const line = `${indent}${n} := ${head(x)}`
        const kids = x.srcs.map((s) => render(s, `${indent}  `)).join("\n")
        return kids ? `${line}\n${kids}` : line
      }
      const line = `${indent}${head(x)}`
      const kids = x.srcs.map((s) => render(s, `${indent}  `)).join("\n")
      return kids ? `${line}\n${kids}` : line
    }

    return render(u)
  },

  topo: (u:UOp):UOp[] =>{
    const out: UOp[] = []
    const seen = new Set<UOp>()
    const active = new Set<UOp>()

    const visit = (x: UOp) => {
      if (seen.has(x)) return
      if (active.has(x)) throw new Error("uop.topo cycle detected")
      active.add(x)
      x.srcs.forEach(visit)
      active.delete(x)
      seen.add(x)
      out.push(x)
    }

    visit(u)
    return out
  },

  dedup: (u: UOp): UOp => {
    const memo = new Map<UOp, UOp>()
    const pool = new Map<string, UOp>()
    const ids = new Map<UOp, number>()
    let nextId = 0

    const getId = (x: UOp): number => {
      if (!ids.has(x)) ids.set(x, nextId++)
      return ids.get(x)!
    }

    const canon = (x: UOp): UOp => {
      const hit = memo.get(x)
      if (hit) return hit

      const srcs = x.srcs.map(canon)
      const args = Object.fromEntries(
        Object.entries(x).filter(([k]) => k !== "srcs")
      )

      const key = JSON.stringify({
        op: x.op,
        args,
        srcs: srcs.map(getId),
      })

      const pooled = pool.get(key)
      if (pooled) {
        memo.set(x, pooled)
        return pooled
      }

      const node = { ...x, srcs } as UOp
      pool.set(key, node)
      memo.set(x, node)
      return node
    }

    return canon(u)
  },

  hash: (u: UOp, _opts: { allowBuffers?: boolean } = {}): string => {
    const memo = new Map<UOp, string>()

    const stable = (v: unknown): string => {
      if (v === null || typeof v !== "object") return JSON.stringify(v)
      if (Array.isArray(v)) return `[${v.map(stable).join(",")}]`
      const obj = v as Record<string, unknown>
      const keys = Object.keys(obj).sort()
      return `{${keys.map((k) => `${JSON.stringify(k)}:${stable(obj[k])}`).join(",")}}`
    }

    const fnv1a = (s: string): string => {
      let h = 0x811c9dc5
      for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i)
        h = Math.imul(h, 0x01000193)
      }
      return (h >>> 0).toString(16).padStart(8, "0")
    }

    const go = (x: UOp): string => {
      const hit = memo.get(x)
      if (hit) return hit

      if (x.op === "BUFFER") {
        const out = fnv1a(`BUFFER|slot:${x.arg.slot}|size:${x.arg.size}`)
        memo.set(x, out)
        return out
      }

      const args = Object.fromEntries(
        Object.entries(x).filter(([k]) => k !== "srcs" && k !== "op")
      )
      const srcHashes = x.srcs.map(go)
      const out = fnv1a(`${x.op}|${stable(args)}|[${srcHashes.join(",")}]`)
      memo.set(x, out)
      return out
    }

    return go(u)
  },

  shape: (u: UOp): number[] => {
    if (u.op === "VIEW") return [...u.views[0]!.dims]
    if (u.op === "CONST") return [u.val.length]
    if (u.op === "BUFFER") return [u.arg.size]
    if (u.op === "RAND") return [u.arg.size]
    if (u.op === "RANGE") return [u.max]
    if (u.op === "SPECIAL") return [u.extent]

    if (u.op === "INDEX") return uop.shape(u.srcs[1])

    if (u.op === "ADD" || u.op === "MUL") return uop.shape(u.srcs[0])
    if (u.op === "NOOP") return uop.shape(u.srcs[0])
    if (u.op === "STORE") return uop.shape(u.srcs[1])

    if (u.op === "REDUCE_AXIS") {
      const s = uop.shape(u.srcs[0])
      return s.filter((_, i) => !u.axis.includes(i))
    }

    if (u.op === "RESHAPE" || u.op === "EXPAND" || u.op === "PERMUTE" || u.op === "PAD" || u.op === "SHRINK") {
      throw new Error(`uop.shape: movement op must be rewritten to VIEW first: ${u.op}`)
    }

    if (u.op === "KERNEL" || u.op === "LINEAR" || u.op === "PROGRAMM" || u.op === "DEFINE_REG" || u.op === "ENDRANGE" || u.op === "REDUCE") {
      throw new Error(`uop.shape: unsupported op ${u.op}`)
    }

    throw new Error(`uop.shape: unknown op ${(u as UOp).op}`)
  },

}
