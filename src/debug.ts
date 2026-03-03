


let debug_val = 0

export const DEBUG = {
  set : (val:number) => debug_val = val,
  get: ()=>debug_val
}

export const debug = (level:number, ...args:any[])=>{
  if (level>=debug_val) console.log(...args)
}