<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# NVFP4

NVFP4 is NVIDIA's 4-bit floating-point format. It stores weights in E2M1, with two values per byte. It has two levels
of scale. Each group of 16 values has an FP8 (E4M3) scale, and each tensor has one FP32 global scale.

[`NVFP4Config`] loads NVFP4 checkpoints as they ship. It also quantizes full-precision checkpoints on the fly.
Checkpoints declare the format as `quant_method="nvfp4"`. See [Fine-grained](./finegrained) for the requirements,
loading options and parallelism shared by every block-scaled format.

Pass [`NVFP4Config`] to quantize a full-precision checkpoint as it loads. The saved model reloads as NVFP4 without a
config.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, NVFP4Config

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype="auto", device_map="auto", quantization_config=NVFP4Config()
)
model.save_pretrained("qwen3-8b-nvfp4")

model = AutoModelForCausalLM.from_pretrained("qwen3-8b-nvfp4", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("NVFP4 is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Activation formats

`activation_format` picks the format the activations are quantized to before each matmul. To change it for a
pre-quantized checkpoint, pass the config with the `activation_format` you want. The weights still load as they ship.

| `activation_format` | Scheme | Activations |
|---|---|---|
| `None` (default) or `"nvfp4"` | W4A4 | NVFP4, against the checkpoint's calibrated global scale when it ships one, on the NVFP4 MMA of Blackwell GPUs |
| `"bf16"` | W4A16 (weight-only) | kept in full precision |

A weight-only checkpoint ships no activation scales. A config passed with it must set `activation_format="bf16"`.
