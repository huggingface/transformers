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
*This model was contributed to Hugging Face Transformers on 2026-08-05.*

# TinyModel

[TinyModel](https://github.com/noanabeshima/tinymodel) is a compact decoder-only language model trained on TinyStories.
It uses learned absolute positions, bias-free query/key/value projections, a biased attention output projection, ReLU
feed-forward layers, and two residual connections per decoder block. It does not use normalization, dropout, tied
embeddings, or key/value caching.

The original checkpoints are distributed as PyTorch `.pt` state dictionaries. The converter supports all three files
from the pinned [checkpoint revision](https://huggingface.co/noanabeshima/tiny_model/tree/502a1f2453f61260c937f7807a1270a167faba07):

| Checkpoint | Layers | SHA-256 |
|---|---:|---|
| `tiny_model.pt` | 4 | `dec406b1ad94cb345b2606d7f8cffa7c1114fcb60850e949eb17274cec30a8c3` |
| `tiny_model_2L_1E.pt` | 2 | `04e8df0cd677a7060558e5c9eb3aaa30dbfe84e4ecc92bf17ef0e405dcf33baf` |
| `tiny_model_2L_3E.pt` | 2 | `26dfc06da85d0e5d4de51a2e90108f9d585a81677bf4bac0ac079e780fda31f4` |

Convert one to a standard Transformers checkpoint before loading it:

```bash
python -m transformers.models.tiny_model.convert_tiny_model_weights_to_hf \
    --checkpoint_path /path/to/tiny_model.pt \
    --output_dir /path/to/tiny-model-hf \
    --expected_num_hidden_layers 4
```

The two-layer checkpoints use `--expected_num_hidden_layers 2`. The converter performs strict source and target key
validation, preserves the published bfloat16 tensors, and writes `config.json` and `model.safetensors`.

```python
import torch

from transformers import TinyModelForCausalLM


model = TinyModelForCausalLM.from_pretrained("/path/to/tiny-model-hf", dtype=torch.bfloat16)
input_ids = torch.tensor([[9996, 51, 56, 4, 36]])

with torch.no_grad():
    logits = model(input_ids).logits
    generated_ids = model.generate(input_ids, max_new_tokens=8, do_sample=False)
```

The original implementation returns log-probabilities. [`TinyModelForCausalLM`] follows the Transformers convention
and returns raw logits. For a numerical comparison, load the native model with `dtype=torch.float32` and
`attn_implementation="sdpa"`, then apply `torch.nn.functional.log_softmax(logits, dim=-1)`. Exact source-equation
parity applies to contiguous, unmasked token sequences with the default positions. The native `attention_mask` and
explicit `position_ids` arguments are standard Transformers API extensions.

> [!NOTE]
> The original text pipeline applies custom normalization and two token-ID remapping tables after a TinyStories GPT-2
> tokenizer. Those tokenizer assets are not present in the checkpoint repository, so this model integration accepts
> token IDs directly and does not substitute a tokenizer with incompatible IDs.

## TinyModelConfig

[[autodoc]] TinyModelConfig

## TinyModel

[[autodoc]] TinyModel
    - forward

## TinyModelForCausalLM

[[autodoc]] TinyModelForCausalLM
    - forward
