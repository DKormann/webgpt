import { Raw } from "./types"

export const log =<T> (...args:[...any[], T]):T =>{
  console.log(...args.map(x=>{
    if (typeof x == "object"){
      if ("__repr__" in x) return x.__repr__()
      return JSON.stringify(x, null, 2)
    }
    return x
  }))
  return args[args.length-1]
}

export const stridesFor = (shape: number[]): number[] =>
  shape.map((_, i) => shape.slice(i + 1).reduce((a, c) => a * c, 1));

export const asShape = (shape:number[], raw:number[]): Raw =>{

  if (shape.length == 0) return raw[0]
  let d = shape[0]
  let n = numel(shape.slice(1,))
  return Array.from({length:d}, ()=>0).map((_,i)=> asShape(shape.slice(1), raw.slice(i*n, (i+1)*n)))
}

export const numel = (shape: number[]): number => shape.reduce((a, b) => a * b, 1);

export const partition =<T> (ls:T[], f:(t:T)=>boolean):[T[],T[]] =>{
  let a : T[] = []
  let b : T[] = []
  ls.forEach(x=>(f(x)?a:b).push(x))
  return [a,b]
}


export const zip =<T, U> (s:T[], ...t:U[][]) => s.map((se,i) => [se, ...t.map(x=> x[i])]) as [T,...U[]][]