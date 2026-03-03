


import {describe, expect, test} from "bun:test"
import { WEBGPU } from "./webgpu"
import { LowGraph, UOp } from "./types"
import { uop } from "./uops"



describe("low level webgpu test",()=>{

  const Device = WEBGPU
  test("create buffer",async ()=>{
    let buffer = Device.createBuffer(10)
    expect((await buffer.read()).length).toEqual(10)
  })

  test("buffer store",async ()=>{
    let buf = Device.createBuffer(10)
    let kernel = Device.createKernel(
      [uop.store(
        uop.const (22),
        uop.buffer(buf),
        uop.const(0)
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
      uop.store( range, buf, range),
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
      uop.store( uop.const(0), buf, uop.const(0)),
      range,
      uop.store( uop.add(uop.index(buf, uop.const(0)), range), buf, uop.const(0)),
      uop.endrange(range),
    ], [buffer])

    await kernel.launch()
    expect((await buffer.read())).toEqual([45])

  })


})
