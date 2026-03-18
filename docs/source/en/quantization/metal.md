<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Metal

Metal quantization performs affine quantization on Apple Silicon (MPS) devices using Metal kernels hosted on the Hugging Face Hub ([kernels-community/mlx-quantization-metal-kernels](https://huggingface.co/kernels-community/mlx-quantization-metal-kernels)). These kernels originate from the [MLX](https://github.com/ml-explore/mlx) framework and are compiled via the [`kernels`](https://github.com/huggingface/kernels) library.

Weights are packed into `uint32` tensors with per-group scales and biases, and the forward pass uses a fused dequantization + matmul Metal kernel (`affine_qmm_t`). This keeps memory usage low while running inference entirely on the GPU with no CPU round-trips.

Supported bit-widths are **2, 4, and 8**. Group size is configurable (default 64).

## Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4) with MPS support
- The `kernels` package:

```bash
pip install kernels
```

The Metal kernels are downloaded from the Hub automatically on first use — no manual compilation required.

## Quantize on-the-fly

Load any model and quantize it during loading by passing a [`MetalConfig`]. All eligible `nn.Linear` layers are replaced with quantized versions.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, MetalConfig

quantization_config = MetalConfig(bits=4, group_size=64)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    device_map="mps",
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
inputs = tokenizer("Apple Silicon is", return_tensors="pt").to("mps")
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Load a pre-quantized model

If a checkpoint already contains quantized weights (`weight` as packed uint32, `scales`, `qbiases`), they are loaded directly — no re-quantization needed.

```python
from transformers import AutoModelForCausalLM, MetalConfig

model = AutoModelForCausalLM.from_pretrained(
    "your-org/model-metal-4bit",
    device_map="mps",
)
```

## Dequantize

On machines without MPS, a pre-quantized checkpoint is automatically dequantized back to float so the model remains usable on CPU or CUDA. You can also force this behavior explicitly:

```python
from transformers import AutoModelForCausalLM, MetalConfig

config = MetalConfig(dequantize=True)
model = AutoModelForCausalLM.from_pretrained(
    "your-org/model-metal-4bit",
    quantization_config=config,
    device_map="cpu",
)
```

## Exclude layers

Certain layers (e.g., `lm_head`) can be excluded from quantization via `modules_to_not_convert`:

```python
config = MetalConfig(bits=4, group_size=64, modules_to_not_convert=["lm_head"])
```

## Configuration options

| Parameter | Default | Description |
|---|---|---|
| `bits` | `4` | Bit-width for weight quantization (2, 4, or 8) |
| `group_size` | `64` | Number of elements per quantization group |
| `modules_to_not_convert` | `None` | List of module names to keep in full precision |
| `dequantize` | `False` | Force dequantization to float (for non-MPS devices) |
