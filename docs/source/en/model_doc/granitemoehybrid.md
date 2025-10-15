<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-05-02 and added to Hugging Face Transformers on 2025-05-05 and contributed by [SukritiSharma](https://huggingface.co/SukritiSharma) and [abrooks9944](https://huggingface.co/abrooks9944).*

# GraniteMoeHybrid

[GraniteMoeHybrid](https://www.ibm.com/new/announcements/ibm-granite-4-0-tiny-preview-sneak-peek) is the smallest upcoming model in the Granite 4.0 family, designed for extreme efficiency on consumer-grade GPUs. It uses a new hybrid Mamba-2/Transformer mixture-of-experts (MoE) architecture with 7B total parameters but only 1B active during inference, enabling reduced memory use while supporting long contexts up to 128K tokens. Despite being only partially trained on 2.5T tokens (of a planned 15T+), it already matches Granite 3.3 2B Instruct and is expected to rival the 8B model after full training. The model is open-sourced on Hugging Face under Apache 2.0 and optimized for concurrent sessions on affordable hardware, making large-context LLM experimentation more accessible.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="ibm-granite/granite-4.0-tiny-preview", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-tiny-preview")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-4.0-tiny-preview", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- [`GraniteMoeHybridForCausalLM`] supports padding-free training. This concatenates distinct training examples while processing inputs as separate batches. Expect ~2x inference acceleration (varies by model and data distribution). Memory usage drops when examples have varying lengths since you avoid padding token overhead.

- Padding-free training requires the `flash-attn`, `mamba-ssm`, and `causal-conv1d` packages. Pass these arguments alongside `input_ids` and `labels`:

  - `position_ids`: `torch.LongTensor` - position index of each token in each sequence
  - `seq_idx`: `torch.IntTensor` - index of each sequence in the batch
  - FlashAttentionKwargs:
    - `cu_seq_lens_q`: `torch.LongTensor` - cumulative sequence lengths of all queries
    - `cu_seq_lens_k`: `torch.LongTensor` - cumulative sequence lengths of all keys
    - `max_length_q`: `int` - longest query length in the batch
    - `max_length_k`: `int` - longest key length in the batch

- Don't provide `attention_mask` inputs. The [`DataCollatorWithFlattening`] generates these arguments automatically when you set `return_seq_idx=True` and `return_flash_attn_kwargs=True`. See the [Improving Hugging Face Training Efficiency Through Packing with Flash Attention](https://huggingface.co/blog/packing-flash-attention) blog post for additional information.

## GraniteMoeHybridConfig

[[autodoc]] GraniteMoeHybridConfig

## GraniteMoeHybridModel

[[autodoc]] GraniteMoeHybridModel
    - forward

## GraniteMoeHybridForCausalLM

[[autodoc]] GraniteMoeHybridForCausalLM
    - forward
