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
*This model was released on {release_date} and added to Hugging Face Transformers on 2024-03-15 and contributed by [saurabhdash](https://huggingface.co/saurabhdash) and [ahmetustun](https://huggingface.co/ahmetustun).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Command-R

[Command-R](https://huggingface.co/papers/2310.06664) is a language model engineered for high-throughput, low-latency retrieval-augmented generation (RAG) and tool use at enterprise scale. It supports a 128,000-token context window, enabling it to reason over very long documents or dialogues, and integrates with external APIs/tools to automate multi-step tasks. The model is optimized for production usage (with strong performance per compute), and fine-tuning of Command R is emphasized as a cost-efficient way to specialize it further.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="CohereLabs/c4ai-command-r-v01", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("CohereLabs/c4ai-command-r-v01")
tokenizer = AutoTokenizer.from_pretrained("CohereLabs/c4ai-command-r-v01")

messages = [{"role": "user", "content": "How do plants generate energy?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

outputs = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.3,)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- Don't use the `dtype` parameter in [`~AutoModel.from_pretrained`] with FlashAttention-2. It only supports `fp16` or `bf16`. Use [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html), set `fp16` or `bf16` to `True` with [`Trainer`], or use [torch.autocast](https://pytorch.org/docs/stable/amp.html#torch.autocast).

## CohereConfig

[[autodoc]] CohereConfig

## CohereTokenizerFast

[[autodoc]] CohereTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary

## CohereModel

[[autodoc]] CohereModel
    - forward

## CohereForCausalLM

[[autodoc]] CohereForCausalLM
    - forward

