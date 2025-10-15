<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-12-18 and added to Hugging Face Transformers on 2024-12-19 and contributed by [ani300](https://github.com/ani300) and [fabianlim](https://github.com/fabianlim).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Bamba

[Bamba-9B](https://github.com/state-spaces/mamba) is a new hybrid language model that combines Mamba2 and Transformer layers to improve inference efficiency. By interleaving Mamba2 layers, it avoids the memory bottleneck of the Transformer’s growing KV-cache, achieving up to 2.5× higher throughput and 2× lower latency in vLLM. The model has 9 billion parameters and was trained on 2.2 trillion tokens of open data, with full training recipes and checkpoints released for reproducibility. It integrates seamlessly with Hugging Face tools like Transformers, TRL, vLLM, and llama.cpp, and comes with additional resources such as a stateless shuffle dataloader and quantization support. Developed in collaboration with IBM, Princeton, CMU, and UIUC, Bamba is intended as an open, efficient foundation for experimenting with hybrid architectures.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="ibm-fms/Bamba-9B", dtype="auto")
pipeline("Plants generate energy through a process known as  ")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ibm-fms/Bamba-9B", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("ibm-fms/Bamba-9B")

inputs = tokenizer("Plants generate energy through a process known as  ", return_tensors='pt', return_token_type_ids=False)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

## Usage tips

- Bamba supports padding-free training. This concatenates distinct training examples while processing inputs as separate batches. Expect ~2x inference acceleration (varies by model and data distribution). Memory usage drops when examples have varying lengths since you avoid padding token overhead.

- Padding-free training requires the flash-attn, mamba-ssm, and causal-conv1d packages. Pass these arguments alongside `input_ids` and `labels`:

- `position_ids`: `torch.LongTensor` - position index of each token in each sequence
- `seq_idx`: `torch.LongTensor` - index of each sequence in the batch
- `FlashAttentionKwargs`:
  - `cu_seq_lens_q`: `torch.LongTensor` - cumulative sequence lengths of all queries
  - `cu_seq_lens_k`: `torch.LongTensor` - cumulative sequence lengths of all keys  
  - `max_length_q`: `int` - longest query length in the batch
  - `max_length_k`: `int` - longest key length in the batch

- Don't provide `attention_mask` inputs. The [`DataCollatorWithFlattening`] generates these arguments automatically when you set `return_seq_idx=True` and `return_flash_attn_kwargs=True`. See the [Improving Hugging Face Training Efficiency Through Packing with Flash Attention](https://huggingface.co/blog/packing-with-FA2) blog post for details.

## BambaConfig

[[autodoc]] BambaConfig

## BambaModel

[[autodoc]] BambaModel
    - forward

## BambaForCausalLM

[[autodoc]] BambaForCausalLM
    - forward
