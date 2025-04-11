# Quantization Concepts

Quantization is a technique used to reduce the memory footprint and computational cost of machine learning models, particularly large ones like those found in the 🤗 Transformers library. It achieves this by representing the model's weights and/or activations using lower-precision data types (like 8-bit integers - `int8`) instead of the standard 32-bit floating-point (`float32`).

## Why Quantize?

Reducing the precision of a model offers several significant advantages:

1.  **Reduced Model Size:** Lower-precision data types require less storage space. An `int8` model, for example, is roughly 4 times smaller than its `float32` counterpart.
2.  **Faster Inference:** Operations on lower-precision data types, especially integers, can be significantly faster on compatible hardware (CPUs and GPUs often have specialized instructions for `int8` operations). This leads to lower latency.
4.  **Reduced Energy Consumption:** Faster computations and smaller memory transfers often translate to lower power usage.

## How Quantization Works (Focus on `int8`)

The core idea is to map the range of values found in the original `float32` weights and activations to the much smaller range representable by `int8` (typically $[-128, 127]$).
<img width="606" alt="quant_visual" src="https://gist.github.com/user-attachments/assets/b45773d8-762a-4223-b1b4-bc15456ca64e" />

### Affine Quantization Scheme

The most common method is the *affine quantization scheme*. For a given `float32` tensor (like a layer's weights), we find its minimum $val_{min}$ and maximum $val_{max}$ values. We then map this range $[val_{min}, val_{max}]$ to the `int8` range $[q_{min}, q_{max}]$, typically $[-128, 127]$.

This mapping is defined by two parameters:

1. **Scale ($S$)**: A positive `float32` number representing the ratio between the `float32` range and the `int8` range.

$$
S = \frac{val_{max} - val_{min}}{q_{max} - q_{min}}
$$

2. **Zero-Point ($Z$)**: An `int8` value that corresponds to the `float32` value $0.0$.

$$
Z = q_{min} - round\left(\frac{val_{min}}{S}\right)
$$

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

### Symmetric vs. Asymmetric Quantization

- **Asymmetric (Affine):** Uses both scale $S$ and zero-point $Z$. This is more flexible, especially if the data range isn't centered around zero.
- **Symmetric:** Assumes the `float32` range is symmetric around zero, i.e., $[ -a, a ]$, and sets $Z = 0$. This simplifies computations and can be faster, though it's less accurate if the data is not symmetric.

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

These are unified under the `HfQuantizer` API and associated `QuantizationConfig` classes.

### Typical Workflow

1. Choose a quantization method suitable for your hardware and use-case (see the [Overview table](./overview)).
2. Load a pre-quantized model from the Hugging Face Hub, or loading a `float32`/`float16`/`bfloat16` model and applying quantization using a specific `QuantizationConfig`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "facebook/opt-350m"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Further Learning

To explore quantization more deeply, check out these short courses created in partnership with DeepLearning.AI:

- [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
- [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth)