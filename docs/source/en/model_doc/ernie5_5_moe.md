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
*This model was contributed to Hugging Face Transformers on 2026-11-15.*

# ERNIE 5.5 MoE

## Overview

ERNIE 5.5 Flash is a text-only mixture-of-experts language model from Baidu, with 280B total and 14B active
parameters. The base checkpoint is [baidu/ERNIE-5.5-Flash-Base-BF16](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base-BF16).

The architecture builds on [DeepSeek-V4](./deepseek_v4) and [DeepSeek-V3.2](./deepseek_v32):

- **Hybrid attention.** Most layers use Heavily Compressed Attention (HCA, from DeepSeek-V4): a shared-KV
  multi-query attention over a 128-token sliding window, plus one compressed KV entry per 128 tokens of the whole
  context. Every seventh layer instead uses Multi-head Latent Attention with DeepSeek Sparse Attention (MLA + DSA, from
  DeepSeek-V3.2): each query attends to its sliding window and to the 2048 older tokens picked by a lightning indexer.
  Both attention types have a per-head attention sink, a low-rank head mix ("Virtual Head Attention") and a sigmoid
  output gate computed from the query latent.
- **Manifold-constrained Hyper-Connections (mHC).** The residual stream is split into 4 parallel streams, mixed in and
  out of every block by learned, Sinkhorn-normalized weights, as in DeepSeek-V4.
- **Latent MoE.** Tokens are projected to a 2048-wide latent space before the 512 routed experts (10 active per token)
  and back up afterwards. Routing sums the scores of two sigmoid routers and scales each selected expert by its own
  learned factor. A shared expert works on the full-width hidden states. The feed-forward blocks use SiTU-GLU, a
  GLU whose gate and linear branches are softly saturated with `tanh`.
- **MultiMax.** The logits go through a learned piece-wise correction before the softmax.

The checkpoint also contains an ECHO multi-token-prediction layer (`model.layers.43`), which is not loaded.

Tips:

- Only the eager attention implementation is supported: the HCA heads are 512-wide, both attention types use
  attention sinks, and both extend or rewrite the attention mask inside the layer.
- The model weights are saved back in the released format, so a fine-tuned model can be served with the same
  inference engines as the original checkpoint.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "baidu/ERNIE-5.5-Flash-Base-BF16"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True))
```

## Ernie5_5_MoeConfig

[[autodoc]] Ernie5_5_MoeConfig

## Ernie5_5_MoeModel

[[autodoc]] Ernie5_5_MoeModel
    - forward

## Ernie5_5_MoeForCausalLM

[[autodoc]] Ernie5_5_MoeForCausalLM
    - forward
