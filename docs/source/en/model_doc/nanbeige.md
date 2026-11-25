<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2026-07-27 and contributed to Hugging Face Transformers on 2026-11-25.*


# Nanbeige

[Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) is a causal language model for reasoning, chat, and tool
use. Its looped Transformer architecture runs the same decoder layers repeatedly, sharing their parameters across loops.

The released checkpoint has 22 decoder layers and runs them twice with `num_loops=2`. It uses grouped-query attention with
48 query heads and 8 key/value heads, a hidden size of 3072, and a maximum context length of 262,144 tokens. Each loop
maintains its own key/value cache for each layer. With the checkpoint's `skip_loop_final_norm=False` setting, the final
RMS normalization is applied after each loop.

See the [technical report](https://huggingface.co/papers/2607.22083) for training and evaluation details.

## Usage

Load the tokenizer and model with the auto classes. This example disables thinking in the chat template to request a short response.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "Nanbeige/Nanbeige4.2-3B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", dtype="auto")

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Which number is larger, 9.11 or 9.8?"}],
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

The tokenizer enables thinking by default. Set `enable_thinking=True` to generate a reasoning trace. For multi-turn
conversations, `preserve_thinking` controls whether reasoning from earlier assistant turns remains in the prompt.

## NanbeigeConfig

[[autodoc]] NanbeigeConfig

## NanbeigeModel

[[autodoc]] NanbeigeModel
    - forward

## NanbeigeForCausalLM

[[autodoc]] NanbeigeForCausalLM
    - forward
