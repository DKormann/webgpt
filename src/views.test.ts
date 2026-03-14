

import { describe, expect, test } from "bun:test";
import { mergeView } from "./kernelize";
import { View } from "./types";
import { contiguos } from "./helpers";



describe("merges views successfully",()=>{

  let mk_cont = (...dims:number[]): View => contiguos(dims)
  let cont_10_10 :View = mk_cont(10,10)
  let view_d10_10_s1_10: View = [
    {stride: 1, size: 10},
    {stride: 10, size: 10},
  ]
  let cont_100 = mk_cont(100)
  let cont_10_10_2 = mk_cont(10,10,2)
  let cont_100_2 = mk_cont(100,2)

  test("cont",()=>{

    expect(cont_100_2)
    .toEqual([{stride:2, size:100}, {stride:1,size:2}])

    expect(cont_10_10_2)
    .toEqual([
      {stride:20,size:10},
      {stride:2,size:10},
      {stride:1,size:2},
    ])

  })

  test("nothing", ()=>{
    expect(mergeView(cont_10_10, cont_10_10))
    .toEqual([cont_10_10])
  })

  test("permute1",()=>{
    expect(mergeView(cont_10_10, view_d10_10_s1_10))
    .toEqual([view_d10_10_s1_10])
  })

  test("double transpose",()=>{
    expect(mergeView(view_d10_10_s1_10, view_d10_10_s1_10))
    .toEqual([cont_10_10])
  })

  test("merge flat", ()=>{
    expect(mergeView(cont_10_10, cont_100))
    .toEqual([cont_10_10])
  })
  
    test("merge 10 10 2 flat", ()=>{
      expect(mergeView(cont_10_10_2, cont_100_2))
      .toEqual([cont_10_10_2])
    })

  test("merge flat 10 10", ()=>{
    expect(mergeView(cont_100, cont_10_10))
    .toEqual([cont_100])
  })

})
