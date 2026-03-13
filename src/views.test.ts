

import { describe, expect, test } from "bun:test";
import { mergeView } from "./kernelize";
import { View } from "./types";
import { stridesFor } from "./helpers";



describe("merges views successfully",()=>{

  let mk_cont = (...dims:number[]): View => ({dims,  strides: stridesFor(dims)})
  let cont_10_10 :View = mk_cont(10,10)
  let perm_10_10: View = {dims:[10,10], strides: [1,10]}
  let flat_100 = mk_cont(100)
  let cont_10_10_2 = mk_cont(10,10,2)
  let cont_100_2 = mk_cont(100,2)

  test("permute1",()=>{
    expect(mergeView(cont_10_10, perm_10_10))
    .toEqual([perm_10_10])
  })
  test("merge flat", ()=>{
    expect(mergeView(cont_10_10, flat_100))
    .toEqual([cont_10_10])
  })

  test("merge 10 10 2 flat", ()=>{
    expect(mergeView(cont_10_10_2, cont_100_2))
    .toEqual([cont_10_10_2])
  })

})
