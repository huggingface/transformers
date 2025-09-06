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
*This model was released on 2025-05-02 and added to Hugging Face Transformers on 2025-05-05.*

# GraniteMoeHybrid

## Overview


The [GraniteMoeHybrid](https://www.ibm.com/new/announcements/ibm-granite-4-0-tiny-preview-sneak-peek) model builds on top of GraniteMoeSharedModel and Bamba. Its decoding layers consist of state space layers or MoE attention layers with shared experts. By default, the attention layers do not use positional encoding.


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "ibm-granite/granite-4.0-tiny-preview"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()

# change input text as desired
prompt = "Write a code to find the maximum value in a list of numbers."

# tokenize the text
input_tokens = tokenizer(prompt, return_tensors="pt")
# generate output tokens
output = model.generate(**input_tokens, max_new_tokens=100)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# loop over the batch to print, in this example the batch size is 1
for i in output:
    print(i)
```

This HF implementation is contributed by [Sukriti Sharma](https://huggingface.co/SukritiSharma) and [Alexander Brooks](https://huggingface.co/abrooks9944).

## Notes

- `GraniteMoeHybridForCausalLM` supports padding-free training which concatenates distinct training examples while still processing inputs as separate batches. It can significantly accelerate inference by [~2x](https://github.com/huggingface/transformers/pull/35861#issue-2807873129) (depending on model and data distribution) and reduce memory-usage if there are examples of varying lengths by avoiding unnecessary compute and memory overhead from padding tokens.

  Padding-free training requires the `flash-attn`, `mamba-ssm`, and `causal-conv1d` packages and the following arguments must be passed to the model in addition to `input_ids` and `labels`.

  - `position_ids: torch.LongTensor`: the position index of each token in each sequence.
  - `seq_idx: torch.IntTensor`: the index of each sequence in the batch.
  - Each of the [`FlashAttentionKwargs`]
    - `cu_seq_lens_q: torch.LongTensor`: the cumulative sequence lengths of all queries.
    - `cu_seq_lens_k: torch.LongTensor`: the cumulative sequence lengths of all keys.
    - `max_length_q: int`: the longest query length in the batch.
    - `max_length_k: int`: the longest key length in the batch.

  The `attention_mask` inputs should not be provided. The [`DataCollatorWithFlattening`] programmatically generates the set of additional arguments above using `return_seq_idx=True` and `return_flash_attn_kwargs=True`. See the [Improving Hugging Face Training Efficiency Through Packing with Flash Attention](https://huggingface.co/blog/packing-with-FA2) blog post for additional information.

  ```python
  from transformers import DataCollatorWithFlattening

  # Example of using padding-free training
  data_collator = DataCollatorWithFlattening(
      tokenizer=tokenizer,
      return_seq_idx=True,
      return_flash_attn_kwargs=True
  )
  ```

## GraniteMoeHybridConfig

[[autodoc]] GraniteMoeHybridConfig

## GraniteMoeHybridModel

[[autodoc]] GraniteMoeHybridModel
    - forward

## GraniteMoeHybridForCausalLM

[[autodoc]] GraniteMoeHybridForCausalLM
    - forward
