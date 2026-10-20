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

# MXFP8

MXFP8 is the OCP microscaling FP8 format. It stores weights in 8-bit floating point (`float8_e4m3fn`). Each group of
32 values along a row has one power-of-two scale (UE8M0). On Blackwell, the tensor cores read these group scales
directly in their block-scaled MMA.

[`Mxfp8Config`] loads MXFP8 checkpoints as they ship, such as `MiniMaxAI/MiniMax-M3-MXFP8`. It also quantizes
full-precision checkpoints on the fly. Checkpoints declare the format as `quant_method="mxfp8"`. Group-32 FP8 exports
of [compressed-tensors](./compressed_tensors) with E8M0 scales also load as [`Mxfp8Config`]. See
[Fine-grained](./finegrained) for the requirements, loading options and parallelism shared by every block-scaled
format.

```py
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("MiniMaxAI/MiniMax-M3-MXFP8", dtype="auto", device_map="auto")
```

## Activation formats

`activation_format` picks the format the activations are quantized to before each matmul. To change it for a
pre-quantized checkpoint, pass the config with the `activation_format` you want. The weights still load as they ship.

| `activation_format` | Scheme | Activations |
|---|---|---|
| `None` (default) or `"mxfp8"` | W8A8 | MXFP8 |
| `"mxfp4"` | W8A4 | MXFP4 |
| `"bf16"` | W8A16 (weight-only) | kept in full precision |
