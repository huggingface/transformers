# Quantization Concepts

Quantization is a technique used to reduce the memory footprint and computational cost of machine learning models, particularly large ones like those found in the 🤗 Transformers library. It achieves this by representing the model's weights and/or activations using lower-precision data types (like 8-bit integers - `int8`) instead of the standard 32-bit floating-point (`float32`).

## Why Quantize?

Reducing the precision of a model offers several significant advantages:

1.  **Reduced Model Size:** Lower-precision data types require less storage space. An `int8` model, for example, is roughly 4 times smaller than its `float32` counterpart.
2.  **Faster Inference:** Operations on lower-precision data types, especially integers, can be significantly faster on compatible hardware (CPUs and GPUs often have specialized instructions for `int8` operations). This leads to lower latency.
4.  **Reduced Energy Consumption:** Faster computations and smaller memory transfers often translate to lower power usage.

## How Quantization Works

The core idea is to map the range of values found in the original `float32` weights and activations to the much smaller range representable by `int8` (typically $[-128, 127]$).
<img width="606" alt="quant_visual" src="https://gist.github.com/user-attachments/assets/b45773d8-762a-4223-b1b4-bc15456ca64e" />

### Affine Quantization Scheme

The most common method is the *affine quantization scheme*. For a given `float32` tensor (like a layer's weights), we find its minimum $val_{min}$ and maximum $val_{max}$ values. We then map this range $[val_{min}, val_{max}]$ to the `int8` range $[q_{min}, q_{max}]$, typically $[-128, 127]$.
 There are two main ways to perform this mapping: Symmetric and Asymmetric.

### Symmetric vs. Asymmetric Quantization

The choice between symmetric and asymmetric quantization determines how the `float32` range is mapped to the `int8` range.

- **Symmetric:** Assumes the original `float32` range is symmetric around zero, i.e., $[ -a, a ]$. This range is then mapped symmetrically to the `int8` range, for example, $[-127, 127]$. A key characteristic is that the `float32` value $0.0$ maps directly to the `int8` value $0$. This requires only one parameter, the **scale ($S$)**, to define the mapping. It can simplify computations but might be less accurate if the original data distribution isn't naturally centered around zero.
- **Asymmetric (Affine):** This method does not assume the data is centered around zero. It maps the exact range $[val_{min}, val_{max}]$ from `float32` to the full `int8` range, like $[-128, 127]$. This requires two parameters: a **scale ($S$)** and a **zero-point ($Z$)**. 

### Affine Quantization Formula

Affine quantization is defined by two parameters:

1. **Scale ($S$)**: A positive `float32` number representing the ratio between the `float32` range and the `int8` range.

$$
S = \frac{val_{max} - val_{min}}{q_{max} - q_{min}}
$$

2. **Zero-Point ($Z$)**: An `int8` value that corresponds to the `float32` value $0.0$.

$$
Z = q_{min} - round\left(\frac{val_{min}}{S}\right)
$$

(Note: In symmetric quantization, Z would typically be fixed at 0).

With these parameters, a `float32` value $x$ can be quantized to `int8` ($q$) using:

$$
q = round\left(\frac{x}{S} + Z\right)
$$

An `int8` value $q$ can be dequantized back to approximate `float32` using:

$$
x \approx S \cdot (q - Z)
$$

<img width="974" alt="dequant" src="https://gist.github.com/user-attachments/assets/05c10fc0-ace3-40ff-962c-c4ea0eb49d60" />

During inference, computations like matrix multiplication are performed using the `int8` values ($q$), and the result is then dequantized back to `float32` (often using a higher-precision accumulation type like `int32` internally) before being passed to the next layer.

### Quantization to `int4` and Weight Packing

<img width="606" alt="weight packing" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/weight_packing.png" />

Quantizing to `int4` further reduces the model size and memory usage (halving it compared to `int8`). The same affine or symmetric quantization principles apply, mapping the `float32` range to the 16 possible values representable by `int4` (e.g., $[-8, 7]$ for signed `int4`).

A key aspect of `int4` quantization is **weight packing**. Since most hardware doesn't natively handle 4-bit data types in memory, two `int4` values are typically packed together into a single `int8` byte for storage and transfer. For example, the first value might occupy the lower 4 bits and the second value the upper 4 bits of the byte (e.g., `packed_byte = (val1 & 0x0F) | (val2 << 4)`).

**Why is `int4` beneficial even without native `int4` compute?**

The primary benefit comes from reduced memory bandwidth. Loading the packed `int4` weights (stored as `int8`) from memory (RAM or VRAM) to the compute units is twice as fast as loading `int8` weights. For large models, memory access is often a significant bottleneck. The speed gain from faster data transfer can outweigh the computational overhead of unpacking and dequantizing on the fly, leading to overall faster inference, especially in memory-bound scenarios.

However, `int4` quantization typically results in a larger accuracy drop compared to `int8`. Advanced quantization techniques like GPTQ or AWQ are often necessary to achieve good performance with `int4`.

### Granularity: Per-Tensor vs. Per-Channel/Group

Quantization parameters ($S$ and $Z$) can be calculated for:

- **Per-Tensor:** One set of $S$ and $Z$ for the entire tensor. Simpler but less accurate if data values vary greatly within the tensor.
- **Per-Channel (or Per-Group/Block):** Separate $S$ and $Z$ for each channel or group. More accurate and better performance, at the cost of slightly more complexity and memory.

<img width="625" alt="Granularities" src="https://gist.github.com/user-attachments/assets/a974fddc-dc31-4ca7-997e-93c648bfda23" />

## Common Quantization Techniques

There are two main types of quantization approaches:

1. **Post-Training Quantization (PTQ):** Quantization is applied  *after* the model has been fully trained.
2. **Quantization-Aware Training (QAT):** Quantization effects are simulated *during* training by inserting "fake quantization" ops that simulate the rounding errors of quantization. This lets the model adapt to quantization, and usually results in better accuracy, especially at lower bit-widths.

## Trade-offs

The primary trade-off in quantization is **efficiency vs. accuracy**. Reducing precision saves resources but inevitably introduces small errors (quantization noise). The goal is to minimize this error using appropriate schemes (affine/symmetric), granularity (per-tensor/channel), and techniques (PTQ/QAT) so that the model's performance on its target task degrades as little as possible.

## Quantization in 🤗 Transformers

The 🤗 Transformers library integrates with several quantization backends like:

- `bitsandbytes`
- `auto-gptq`
- `awq`
- And more...

These are unified under the `HfQuantizer` API and associated `QuantizationConfig` classes. You can integrate your own custom quantization backends by implementing a custom `HfQuantizer` and `QuantizationConfig`, allowing seamless use within the 🤗 Transformers ecosystem. For more details,  you can check out this [example](https://github.com/huggingface/transformers/blob/main/examples/quantization/custom_quantization_int8_example.py) and the [contribution guide](./contribute).

### Typical Workflow

1. Choose a quantization method suitable for your hardware and use-case (see the [Overview table](./overview) or [selecting a quantization method doc](./selecting)).
2. Load a pre-quantized model from the Hugging Face Hub, or loading a `float32`/`float16`/`bfloat16` model and applying quantization using a specific `QuantizationConfig`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### FP8 Quantization (A8W8)

<img width="606" alt="fp8" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/fp8.png" />

A newer databyte, 8-bit floating-point (FP8), offers another way to reduce precision while retaining more accuracy than INT8 in certain scenarios. FP8 keeps the floating-point structure (sign, exponent, mantissa) but use fewer bits.

**FP8 Formats:** The two common FP8 variants are:
- **E4M3:** 1 sign bit, 4 exponent bits, 3 mantissa bits. Offers higher precision (more mantissa bits) but a smaller dynamic range (fewer exponent bits).
- **E5M2:** 1 sign bit, 5 exponent bits, 2 mantissa bits. Offers a wider dynamic range but lower precision.

**A8W8 Scheme:** This refers to quantizing both activations (A) and weights (W) to 8-bit precision.

**Hardware Requirement:** While INT8 has broad support, efficient FP8 computation requires specific hardware capabilities found in newer GPUs like NVIDIA H100/H200/B100 and AMD Instinct MI300 series. Without native hardware acceleration, the benefits of FP8 might not be fully realized.

**Integration in 🤗 Transformers:** FP8 support is integrated via specific backends like `FBGEMM`, `FineGrainedFP8`, and `compressed-tensors`. These handle the underlying FP8 conversion and computation when the appropriate hardware and configurations are used.

## Further Learning

To explore quantization and related performance optimization concepts more deeply, check out these resources:

- [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
- [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth)
- [Introduction to Quantization cooked in 🤗 with 💗🧑‍🍳](https://huggingface.co/blog/merve/quantization)
- [EfficientML.ai Lecture 5 - Quantization Part I](https://www.youtube.com/watch?v=RP23-dRVDWM)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [Accelerating Generative AI with PyTorch Part 2: LLM Optimizations](https://pytorch.org/blog/accelerating-generative-ai-2/)