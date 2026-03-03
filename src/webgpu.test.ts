


import {describe, expect, test} from "bun:test"
import { WEBGPU } from "./webgpu"
import { LowGraph, UOp } from "./types"
import { uop } from "./uops"



describe("low level webgpu test",()=>{

  const Device = WEBGPU
  const jsRand = (x: number) => {
    let z = ((x >>> 0) + 0x9e3779b9) >>> 0
    z = Math.imul((z ^ (z >>> 16)) >>> 0, 0x85ebca6b) >>> 0
    z = Math.imul((z ^ (z >>> 13)) >>> 0, 0xc2b2ae35) >>> 0
    z = (z ^ (z >>> 16)) >>> 0
    return z * 2.3283064365386963e-10
  }
  test("create buffer",async ()=>{
    let buffer = Device.createBuffer(10)
    expect((await buffer.read()).length).toEqual(10)
  })

  test("buffer store",async ()=>{
    let buf = Device.createBuffer(10)
    let kernel = Device.createKernel(
      [uop.store(
        uop.const (22),
        uop.index(uop.buffer(buf), uop.const(0))
      ) as LowGraph],
      [buf]
    )

    await kernel.launch()
    expect((await buf.read()).slice(0,1)).toEqual([22])

  })

  test("kernel: range",async ()=>{
    let buffer = Device.createBuffer(10)

    let buf = uop.buffer(buffer)
    let range = uop.range(10)

    let kernel = Device.createKernel([
      range,
      uop.store( range, uop.index(buf, range)),
      uop.endrange(range),
    ],
      [buffer]
    )

    await kernel.launch()
    expect((await buffer.read())).toEqual([0,1,2,3,4,5,6,7,8,9])

  })



  test("kernel: range sum ",async ()=>{
    let buffer = Device.createBuffer(1)

    let buf = uop.buffer(buffer)
    let range = uop.range(10)


    let kernel = Device.createKernel([
      uop.store( uop.const(0), uop.index(buf, uop.const(0))),
      range,
      uop.store( uop.add(uop.index(buf, uop.const(0)), range), uop.index(buf, uop.const(0))),
      uop.endrange(range),
    ], [buffer])

    await kernel.launch()
    expect((await buffer.read())).toEqual([45])

  })

  test("kernel: range rand", async () => {
    const buffer = Device.createBuffer(4)
    const buf = uop.buffer(buffer)
    const range = uop.range(4)
    const seed = 7

    const kernel = Device.createKernel([
      range,
      uop.store({ op: "RAND", srcs: [], seed }, uop.index(buf, range)),
      uop.endrange(range),
    ], [buffer])

    await kernel.launch()
    const out = await buffer.read()
    const expected = [0, 1, 2, 3].map((i) => jsRand((seed ^ i) >>> 0))
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(expected[i], 6)
  })


})
