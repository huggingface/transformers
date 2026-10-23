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

[TinyModel](https://huggingface.co/noanabeshima/tiny_model) is a compact decoder-only language model trained on TinyStories.
It uses learned absolute positions, bias-free query/key/value projections, a biased attention output projection, ReLU
feed-forward layers, and two residual connections per decoder block. It does not use normalization, dropout, tied
embeddings, or key/value caching.

This model was contributed by [sushmanthreddy](https://github.com/sushmanthreddy).
The original code can be found in [noanabeshima/tinymodel](https://github.com/noanabeshima/tinymodel/tree/ae0c9031c467d05e12914bd7be19aaa6262dc7bf).

## Checkpoints

The weights are available in the [noanabeshima/tiny_model Hub repository](https://huggingface.co/noanabeshima/tiny_model).
The converter supports all three files from revision `502a1f2453f61260c937f7807a1270a167faba07`:

| Checkpoint | Layers |
|---|---:|
| [tiny_model.pt](https://huggingface.co/noanabeshima/tiny_model/blob/502a1f2453f61260c937f7807a1270a167faba07/tiny_model.pt) | 4 |
| [tiny_model_2L_1E.pt](https://huggingface.co/noanabeshima/tiny_model/blob/502a1f2453f61260c937f7807a1270a167faba07/tiny_model_2L_1E.pt) | 2 |
| [tiny_model_2L_3E.pt](https://huggingface.co/noanabeshima/tiny_model/blob/502a1f2453f61260c937f7807a1270a167faba07/tiny_model_2L_3E.pt) | 2 |

These are raw PyTorch state dictionaries, rather than Transformers checkpoints with a `config.json`.
Convert a file locally before loading it with [`~PreTrainedModel.from_pretrained`], as shown below.

## Usage

> [!NOTE]
> The original text pipeline uses custom normalization and token-ID remapping assets that are not included with the
> published checkpoints. The example therefore uses token IDs directly. A GPT-2 tokenizer alone does not produce
> compatible IDs, so do not substitute one to encode or decode text.

The following example downloads the four-layer checkpoint, converts it into `./tiny-model-hf`, and loads that
directory for FP16 inference. If your installed Transformers version does not include TinyModel,
[install Transformers from source](../installation#source-install).

```python
import torch
from huggingface_hub import hf_hub_download

from transformers import TinyModelForCausalLM
from transformers.models.tiny_model.convert_tiny_model_weights_to_hf import convert_tiny_model_checkpoint


checkpoint_path = hf_hub_download(
    repo_id="noanabeshima/tiny_model",
    filename="tiny_model.pt",
    revision="502a1f2453f61260c937f7807a1270a167faba07",
)
output_dir = "./tiny-model-hf"
convert_tiny_model_checkpoint(checkpoint_path, output_dir, expected_num_hidden_layers=4)

model = TinyModelForCausalLM.from_pretrained(output_dir, dtype=torch.float16).eval()
input_ids = torch.tensor([[9996, 51, 56, 4, 36]], device=model.device)

with torch.no_grad():
    logits = model(input_ids).logits
    generated_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False, use_cache=False)

print(generated_ids.tolist())
# [[9996, 51, 56, 4, 36, 1, 38, 6, 4, 31, 66, 39, 49, 0, 21]]
```

To use a two-layer checkpoint, change `filename` to `tiny_model_2L_1E.pt` or `tiny_model_2L_3E.pt` and set
`expected_num_hidden_layers=2`. Use a separate output directory for each converted checkpoint.

The converter preserves the released bfloat16 weights and writes `config.json` and `model.safetensors`.
Passing `dtype=torch.float16` to `from_pretrained` casts those weights for inference.

## Usage notes

- The released checkpoints support sequences of up to 256 tokens, including the prompt and generated tokens.
- Generation recomputes the full sequence at each step. Keep `use_cache=False`; `use_cache=True` and
  `past_key_values` are unsupported.
- [`TinyModelForCausalLM`] returns raw logits, while the original implementation returns log-probabilities.
  Apply `torch.nn.functional.log_softmax(logits, dim=-1)` when comparing outputs with the original model.
- `attention_mask` and explicit `position_ids` are supported. For comparisons with the original implementation,
  use contiguous, unmasked token sequences with the default positions.

## TinyModelConfig

[[autodoc]] TinyModelConfig

## TinyModel

[[autodoc]] TinyModel
    - forward

## TinyModelForCausalLM

[[autodoc]] TinyModelForCausalLM
    - forward
