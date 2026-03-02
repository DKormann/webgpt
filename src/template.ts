import { Tensor } from "./tensor.ts";

const info = new Map<string, string>([
  ["const", "Create a filled tensor. ex: Tensor.const(1, [2,3])"],
  ["rand", "Create random tensor in [0,1). ex: Tensor.rand([2,3])"],
  ["new", "Create from nested arrays. ex: Tensor.new([[1,2],[3,4]])"],
  ["linear", "Linear layer helper. ex: Tensor.linear(x, w, b?)"],
  ["add", "Elementwise add. ex: a.add(b)"],
  ["mul", "Elementwise multiply. ex: a.mul(b)"],
  ["matmul", "Matrix multiply (2D). ex: a.matmul(b)"],
  ["reshape", "Change view shape. ex: t.reshape([3,2])"],
  ["permute", "Reorder axes. ex: t.permute([1,0])"],
  ["expand", "Broadcast dim=1 axes. ex: t.expand([2,3])"],
  ["pad", "Add zero padding. ex: t.pad([[1,1],[0,0]])"],
  ["shrink", "Crop ranges. ex: t.shrink([[1,3],[0,2]])"],
  ["sum", "Reduce add over dims. ex: t.sum([1])"],
  ["prod", "Reduce multiply over dims. ex: t.prod([0])"],
  ["run", "Execute tensor async. ex: await t.run('js')"],
  ["backward", "Backpropagate grads. ex: loss.backward()"]
]);

const renderList = (keys: string[]): string =>
  keys
    .sort()
    .map((k) => (info.has(k) ? `${k} - ${info.get(k)}` : k))
    .join("\n");

export const renderTemplate = (app: HTMLElement): void => {
  app.innerHTML = `
    <h1>Tensor Template</h1>
    <p>This project models tensor compute as a UOP graph plus shape/view transforms.</p>
    <p>
      Core concepts:
      <code>UOP</code> nodes for compute, <code>Shape</code> for dims/strides/offset/mask,
      and runtime backends (<code>js</code>, <code>naive</code>, <code>webgpu</code>).
    </p>
    <h2>Tensor Static Methods</h2>
    <pre id="tensor-static"></pre>
    <h2>Tensor Instance Methods</h2>
    <pre id="tensor-dynamic"></pre>
    <h2>Autograd Example</h2>
    <pre><code>const x = Tensor.new([2, 3], { requiresGrad: true });
const y = x.mul(x).sum();   // y = 2^2 + 3^2 = 13
y.backward();               // dy/dx = 2x
const yValue = await y.run();       // 13
const xGrad = await x.grad!.run();  // [4, 6]
return { yValue, xGrad };</code></pre>
    <p><a href="/">Back to playground</a></p>
  `;

  const staticEl = app.querySelector<HTMLPreElement>("#tensor-static");
  const dynamicEl = app.querySelector<HTMLPreElement>("#tensor-dynamic");
  if (staticEl) staticEl.textContent = renderList(Object.keys(Tensor));
  if (dynamicEl) dynamicEl.textContent = renderList(Object.keys(Tensor.new([0])));
};
