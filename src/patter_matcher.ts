import { Op, UOp } from "./types";


type PatternCtx = {[key:string]:UOp}

export class UPat{
  constructor(public name:string = "", public op:Op | null | Op[] = null, public srcs: UPat[] | null = null){}

  _match (u:UOp, ctx:PatternCtx): (null| PatternCtx) {
    if (this.op != null && (!this.op.includes(u.op)) && (this.op != u.op)) return null
    if (this.name in ctx) if (ctx[this.name] !== u) return null
    if (this.name) ctx = {...ctx, [this.name]: u}
    let check = (pats: UPat[]) => pats.reduce((acc: PatternCtx|null, c:UPat, i:number)=> acc == null ? acc : c._match(u.srcs[i], acc), ctx)
    if (this.srcs == null) return ctx
    return check(this.srcs) ?? ((["ADD", "MUL"].includes(u.op)) ?  check(this.srcs.reverse()) : null)
  }

  match (u:UOp) {return this._match(u, {})}
}


export class PatternMatcher {
  constructor(public pats:[UPat, (c:PatternCtx) => (UOp | null)][]){}
  match(graph:UOp){
    let found = true;
    while (found) {
      found = false
      for (let [pat,fn] of this.pats){
        let go = (u:UOp):UOp=>{
          (r=>{if (r){found=true; u = r}})
          ((c=>c && fn(c))(pat.match(u)))
          return {...u, srcs: u.srcs.map(go)} as UOp
        }
        graph = go(graph)
      }
    }
    return graph
  }
}


