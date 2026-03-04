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
