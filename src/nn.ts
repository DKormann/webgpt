import type { Tensor } from "./tensor.ts";

export const matmul = (a: Tensor, b: Tensor): Tensor => {
  if (a.shape.dims.length !== 2 || b.shape.dims.length !== 2) {
    throw new Error("matmul expects 2D tensors");
  }

  const [m, k] = a.shape.dims;
  const [k2, n] = b.shape.dims;
  if (k !== k2) throw new Error(`matmul shape mismatch: [${m},${k}] x [${k2},${n}]`);

  return a
    .reshape([m, k, 1])
    .expand([m, k, n])
    .mul(b.reshape([1, k, n]).expand([m, k, n]))
    .sum([1]);
};

const addBias = (out: Tensor, bias: Tensor): Tensor => {
  if (bias.shape.dims.length === 1) {
    const outFeatures = out.shape.dims[out.shape.dims.length - 1];
    const [biasFeatures] = bias.shape.dims;
    if (outFeatures !== biasFeatures) {
      throw new Error(`bias shape mismatch: expected last dim ${outFeatures}, got ${biasFeatures}`);
    }
    if (out.shape.dims.length === 1) return out.add(bias);
    return out.add(bias.reshape([1, biasFeatures]).expand(out.shape.dims));
  }

  if (bias.shape.dims.length === out.shape.dims.length) {
    const same = bias.shape.dims.every((d, i) => d === out.shape.dims[i]);
    if (!same) throw new Error(`bias shape mismatch: expected [${out.shape.dims}], got [${bias.shape.dims}]`);
    return out.add(bias);
  }

  throw new Error("bias must be 1D (features) or same shape as output");
};

export const linear = (input: Tensor, weight: Tensor, bias?: Tensor): Tensor => {
  if (weight.shape.dims.length !== 2) throw new Error("linear expects weight to be 2D [inFeatures, outFeatures]");
  const [inFeatures, outFeatures] = weight.shape.dims;

  if (input.shape.dims.length === 1) {
    const [features] = input.shape.dims;
    if (features !== inFeatures) {
      throw new Error(`linear shape mismatch: input features ${features} != weight inFeatures ${inFeatures}`);
    }
    // Avoid reshape from [1, outFeatures] -> [outFeatures] which is currently unstable in naive runtime.
    const out = matmul(input.reshape([1, features]), weight).sum([0]);
    return bias ? addBias(out, bias) : out;
  }

  if (input.shape.dims.length === 2) {
    const [, features] = input.shape.dims;
    if (features !== inFeatures) {
      throw new Error(`linear shape mismatch: input features ${features} != weight inFeatures ${inFeatures}`);
    }
    const out = matmul(input, weight);
    return bias ? addBias(out, bias) : out;
  }

  throw new Error("linear expects input to be 1D [features] or 2D [batch, features]");
};
