var G=Object.defineProperty;var w=(r,o)=>{for(var e in o)G(r,e,{get:o[e],enumerable:!0,configurable:!0,set:(t)=>o[e]=()=>t})};var P={};w(P,{webgpuAvailable:()=>b,vectorAdd:()=>C});var u=null,b=()=>typeof navigator<"u"&&!!navigator.gpu,A=async()=>{if(u)return u;return u=(async()=>{if(!b())throw Error("WebGPU unavailable");let r=await navigator.gpu.requestAdapter();if(!r)throw Error("No GPU adapter");return r.requestDevice()})(),u},l=(r)=>new Float32Array(r),C=async(r,o)=>{if(r.length!==o.length)throw Error("vectorAdd length mismatch");let e=await A(),t=r.length,a=t*4,s=l(r),y=l(o),c=e.createBuffer({size:a,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),f=e.createBuffer({size:a,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),d=e.createBuffer({size:a,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),n=e.createBuffer({size:a,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});e.queue.writeBuffer(c,0,s),e.queue.writeBuffer(f,0,y);let U=e.createShaderModule({code:`
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> out: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i < arrayLength(&a)) {
          out[i] = a[i] + b[i];
        }
      }
    `}),p=e.createComputePipeline({layout:"auto",compute:{module:U,entryPoint:"main"}}),v=e.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:c}},{binding:1,resource:{buffer:f}},{binding:2,resource:{buffer:d}}]}),g=e.createCommandEncoder(),i=g.beginComputePass();i.setPipeline(p),i.setBindGroup(0,v),i.dispatchWorkgroups(Math.ceil(t/64)),i.end(),g.copyBufferToBuffer(d,0,n,0,a),e.queue.submit([g.finish()]),await n.mapAsync(GPUMapMode.READ);let B=new Float32Array(n.getMappedRange().slice(0));return n.unmap(),c.destroy(),f.destroy(),d.destroy(),n.destroy(),Array.from(B)};var m=document.querySelector("#app");if(m){m.innerHTML='<h1>WebGPU Demo</h1><p id="out">Running...</p>';let r=document.querySelector("#out");if(r){let{vectorAdd:o,webgpuAvailable:e}=await Promise.resolve().then(() => P);if(!e())r.textContent="WebGPU unavailable in this browser.";else try{let t=[1,2,3,4],a=[10,20,30,40],s=await o(t,a);r.textContent=`vectorAdd([${t}], [${a}]) = [${s}]`}catch(t){r.textContent=`WebGPU error: ${String(t)}`}}}
