<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Quantization concepts

A quantized number can only take one of a small set of values, 256 for int8 and 16 for int4, so every quantization method introduces rounding error. Methods differ in how they keep that error from hurting the model's output. This page explains the choices behind those differences: how numbers are mapped to that small set, which tensors get quantized, which data types are used, and when quantization happens.

If you're new to quantization in Transformers, start with the [Overview](./overview).

## Map values to fewer bits

Quantization maps a range of high-precision values onto the small set of integers a low-bit type can represent. int8 can represent 256 integers, `[-128, 127]`, and int4 can represent 16.

*Affine quantization* is the most common mapping. It takes the minimum and maximum values of a tensor and stretches the integer range over it. In the example below, a tensor whose values run from -1.8 to 2.4 is mapped to int8, and 0.0 lands on -19.

```text
bf16  -1.8             0.0                    2.4
       ├────────────────┼──────────────────────┤
       ▼                ▼                      ▼
int8  -128             -19                    127
```

Two parameters define the mapping.

- The *scale* `S` is the size of one integer step in the original units.

$$
S = \frac{val_{max} - val_{min}}{q_{max} - q_{min}}
$$

- The *zero point* `Z` is the integer that represents `0.0`.

$$
Z = q_{min} - round\left(\frac{val_{min}}{S}\right)
$$

A value `x` is quantized to the integer `q` by dividing by the scale and shifting by the zero point.

$$
q = round\left(\frac{x}{S} + Z\right)
$$

Dequantizing reverses the mapping and gives back an approximation of `x`. The difference between `x` and its approximation is the rounding error.

$$
x \approx S \cdot (q - Z)
$$

In the example above, `S` is about 0.0165 and `Z` is -19. The value 0.5 quantizes to 11, and 11 dequantizes to about 0.494, a rounding error of about 0.006.

```text
┌─────┐  quantize   ┌────┐ dequantize   ┌───────┐
│ 0.5 │ ──────────▶ │ 11 │ ───────────▶ │ 0.494 │
└─────┘             └────┘              └───────┘
```

Affine quantization comes in two forms.

- *Asymmetric* quantization uses the formulas above. It maps the exact range, so no representable values go to waste when the weights aren't centered on zero.
- *Symmetric* quantization assumes the range is `[-a, a]` and fixes `Z` at 0, so only the scale is stored. The math is cheaper, but some representable values go unused when the weights lean to one side of zero.

## Choose the granularity

A single scale for a whole tensor is cheap to store, but one large outlier stretches the range and spreads the representable values farther apart for every other weight. Smaller groups give each scale a narrower range to cover.

- Per-tensor uses one scale, and one zero point, for the entire tensor.
- Per-channel uses one scale for each output channel, which is each row of a weight matrix.
- Per-group, or per-block, uses one scale for each group of consecutive values, such as 128 weights. Each group costs extra storage for its scale.

The diagram below shows the three options on a weight matrix with three rows. Each `s` is one scale.

```text
per-tensor        per-channel           per-group
┌────────────┐    ┌────────────┐        ┌──────┬──────┐
│            │    │            │ s0     │  s0  │  s1  │
│            │    ├────────────┤        ├──────┼──────┤
│     s      │    │            │ s1     │  s2  │  s3  │
│            │    ├────────────┤        ├──────┼──────┤
│            │    │            │ s2     │  s4  │  s5  │
└────────────┘    └────────────┘        └──────┴──────┘
```

Most low-bit methods quantize per group. [GPTQ](./gptq) defaults to groups of 128 weights, and [Fine-grained FP8](./finegrained_fp8) defaults to 128x128 blocks. The FP4 formats in [Data types](#data-types) build the block size into the format itself.

## Quantize weights or activations

Methods differ in which tensors they quantize, and that decides where the matrix multiplication runs.

- *Weight-only* methods store the weights in low precision and leave the activations in bf16 or fp16. Before each matmul, the kernel dequantizes the weights to a higher-precision compute dtype. bitsandbytes 4-bit, AWQ, GPTQ, and HQQ work this way. In bitsandbytes, `bnb_4bit_compute_dtype` sets that compute dtype.
- *Weight-and-activation* methods quantize the activations as well, and the matmul runs directly in low precision. This needs hardware that supports the low-precision matmul, like FP8 on recent GPUs. The scheme is often written as W8A8 for 8-bit weights and activations.

Weight-only methods save memory but don't always run faster, because dequantizing adds work to every matmul. They speed up generation when the model is *memory-bound*. Generating one token at a time, a GPU spends most of each step reading weights from memory, not computing. Reading 4-bit weights moves a quarter of the data that bf16 weights do, and that saving can outweigh the cost of dequantizing.

Weight-and-activation methods help when the model is *compute-bound*, as in prefill or large batches, because the low-precision matmul itself is faster. Some methods switch between the two. NVFP4 runs a 4-bit-weight, 4-bit-activation matmul for prefill and a 4-bit-weight, 16-bit-activation matmul for decoding.

## Data types

The data type sets which values a quantized number can represent.

### int8 and int4

int8 and int4 represent evenly spaced integers and use the affine mapping above. int4 can represent only 16 values, so it loses more accuracy than int8. Methods that use int4 typically quantize per group or use calibration to keep accuracy up.

Most hardware has no 4-bit storage type, so two int4 values are *packed* into one byte. For example, the first value can take the lower 4 bits and the second value the upper 4 bits. Kernels unpack the values before computing with them.

```text
┌──────┐  ┌──────┐        ┌──────┬──────┐
│ val1 │  │ val2 │        │ val2 │ val1 │
│ 0101 │  │ 0011 │   →    │ 0011 │ 0101 │  1 byte
└──────┘  └──────┘        └──────┴──────┘
  int4      int4            high   low
```

Some 4-bit types space their representable values unevenly to match how weights are distributed. bitsandbytes NF4 puts more of its values near zero, where most weights are.

### FP8

FP8 keeps the floating-point layout of a sign, an exponent, and a mantissa in 8 bits. Its representable values are closer together near zero and farther apart at large magnitudes, which suits the spread of values in weights and activations.

FP8 comes in two variants.

- E4M3 has 4 exponent bits and 3 mantissa bits. It's more precise but covers a smaller range. It's the usual choice for weights and activations during inference.
- E5M2 has 5 exponent bits and 2 mantissa bits. It covers a wider range with less precision.

```text
E4M3   ┌─┬───────┬─────┐
       │S│E E E E│M M M│
       └─┴───────┴─────┘
E5M2   ┌─┬─────────┬───┐
       │S│E E E E E│M M│
       └─┴─────────┴───┘
  S sign   E exponent   M mantissa
```

FP8 matmuls need hardware support, such as NVIDIA GPUs with compute capability 8.9 or higher (Ada, Hopper, Blackwell), AMD Instinct MI300, and Intel XPU. Transformers supports FP8 through [Fine-grained FP8](./finegrained_fp8), [FBGEMM FP8](./fbgemm_fp8), and [compressed-tensors](./compressed_tensors).

### FP4 with block scales

FP4 has 1 sign bit, 2 exponent bits, and 1 mantissa bit (E2M1). It represents only 0, 0.5, 1, 1.5, 2, 3, 4, and 6, positive and negative. With so few values, FP4 formats can't work with one scale per tensor. They attach a scale to every small block of values and store the scales as part of the format.

| Format | Block size | Scale per block | Extra scale |
|---|---|---|---|
| [MXFP4](./mxfp4) | 32 values | 8-bit power of two (E8M0) | None |
| [NVFP4](./nvfp4) | 16 values | FP8 | One FP32 scale per tensor |

MXFP4 is a microscaling format. A power-of-two scale only shifts the exponent, so applying it is cheap, but it can only land on powers of two. NVFP4 uses smaller blocks and an FP8 scale that can take values between powers of two. This keeps more accuracy, at the cost of a per-tensor scale that keeps the FP8 block scales in range. [Four Over Six](./fouroversix) and [FP-Quant](./fp_quant) also produce NVFP4 or MXFP4 weights.

The same block idea shows up at 8 bits. MXFP8 stores FP8 values with power-of-two block scales, and [Fine-grained FP8](./finegrained_fp8) loads it.

### Codebooks

Below 4 bits, some methods stop rounding each weight to the nearest representable value. They quantize groups of weights together as *vectors*. A codebook holds a set of learned vectors, and each group of weights is stored as the index of its closest vector. [AQLM](./aqlm) defaults to groups of 8 weights and a 16-bit index, which works out to 2 bits per weight. [VPTQ](./vptq) also uses codebooks. Finding good codebooks takes a long optimization over calibration data, so these models are quantized ahead of time and loaded as checkpoints.

## When quantization happens

A model is quantized either after training or during it. Most methods in Transformers quantize after training.

### Post-training quantization

*Post-training quantization (PTQ)* quantizes a model that's already trained, without changing what it learned. PTQ methods differ in whether they need data, and that decides which group a method falls into in the documentation's table of contents.

- Without calibration data, the scales come from the weights alone, and quantization is fast enough to run while the model loads. bitsandbytes, HQQ, and torchao work this way. These methods are listed under *Quantize while loading*.
- With calibration data, the method runs a small dataset through the model and uses the activations to decide how to quantize each layer. Calibration usually gives better accuracy at 4 bits and below, but it takes minutes to hours.

Calibration methods use the activations in different ways. GPTQ quantizes a layer one column at a time and adjusts the remaining columns to make up for each column's rounding error. AWQ finds the weight channels with the largest activations and scales them up before rounding to protect them.

Where the calibration runs decides how you use the method in Transformers.

- [GPTQ](./gptq) calibrates inside `from_pretrained` when you pass a `dataset` to [`GPTQConfig`], so it's listed under *Quantize while loading*.
- Other calibration methods, like AWQ, AutoRound, AQLM, and VPTQ, calibrate in their own libraries. Transformers loads the result, so they're listed under *Load pre-quantized checkpoints*.

### Quantization-aware training

*Quantization-aware training (QAT)* simulates quantization during training. The forward pass rounds values as quantization would while gradients update the full-precision weights, so the model learns to tolerate the rounding error. QAT usually gives the best accuracy at very low bit widths, but it requires training the model.

[BitNet](./bitnet) models are trained this way, with their 1.58-bit weights in the loop, and Transformers loads the trained checkpoints.

## Resources

- [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
- [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth)
- [Introduction to Quantization cooked in 🤗 with 💗🧑‍🍳](https://huggingface.co/blog/merve/quantization)
- [EfficientML.ai Lecture 5 - Quantization Part I](https://www.youtube.com/watch?v=RP23-dRVDWM)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [Accelerating Generative AI with PyTorch Part 2: LLM Optimizations](https://pytorch.org/blog/accelerating-generative-ai-2/)
