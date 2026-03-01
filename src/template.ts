import { Tensor } from "./tensor.ts";

const staticEl = document.querySelector<HTMLPreElement>("#tensor-static");
const dynamicEl = document.querySelector<HTMLPreElement>("#tensor-dynamic");

const info = new Map<string, string>([
  ["const", "Create a filled tensor. ex: Tensor.const(1, [2,3])"],
  ["new", "Create from nested arrays. ex: Tensor.new([[1,2],[3,4]])"],
  ["add", "Elementwise add. ex: a.add(b)"],
  ["reshape", "Change view shape. ex: t.reshape([3,2])"],
  ["permute", "Reorder axes. ex: t.permute([1,0])"],
  ["expand", "Broadcast dim=1 axes. ex: t.expand([2,3])"],
  ["pad", "Add zero padding. ex: t.pad([[1,1],[0,0]])"],
  ["shrink", "Crop ranges. ex: t.shrink([[1,3],[0,2]])"],
  ["sum", "Reduce add over dims. ex: t.sum([1])"],
  ["prod", "Reduce multiply over dims. ex: t.prod([0])"],
  ["run", "Execute tensor. ex: t.run('js')"]
]);

const renderList = (keys: string[]): string =>
  keys
    .sort()
    .map((k) => (info.has(k) ? `${k} - ${info.get(k)}` : k))
    .join("\n");

if (staticEl) {
  staticEl.textContent = renderList(Object.keys(Tensor));
}

if (dynamicEl) {
  dynamicEl.textContent = renderList(Object.keys(Tensor.new([0])));
}
