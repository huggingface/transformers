<!--Copyright 2025 The LG AI Research and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-07-15 and added to Hugging Face Transformers on 2025-07-26.*

# EXAONE 4

[EXAONE 4.0](https://huggingface.co/papers/2507.11407) combines a Non-reasoning mode and a Reasoning mode to merge the usability of EXAONE 3.5 with the advanced reasoning of EXAONE Deep. It introduces agentic tool-use capabilities and expands multilingual support to include Spanish alongside English and Korean. The model series includes a 32B version for high performance and a 1.2B version for on-device use. EXAONE 4.0 outperforms comparable open-weight models, remains competitive with frontier models, and is publicly available for research on Hugging Face.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="LGAI-EXAONE/EXAONE-4.0-32B", dtype="auto")
pipeline("Plants generate energy through a process known as  ")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("LGAI-EXAONE/EXAONE-4.0-32B", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-32B")

messages = [{"role": "user", "content": "How do plants generate energy?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

outputs = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.3,)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- EXAONE 4.0 models have reasoning capabilities for complex problems. Activate reasoning mode with `enable_thinking=True` in the tokenizer. This opens a reasoning block starting with `<think>` tag without closing it. Model generation with reasoning mode is sensitive to sampling parameters. Check the [Usage Guideline](https://github.com/LG-AI-EXAONE/EXAONE-4.0#usage-guideline) on the official GitHub page for better quality.
- EXAONE 4.0 models work as agents with tool calling capabilities. Provide tool schemas to the model for effective tool calling.

## Exaone4Config

[[autodoc]] Exaone4Config

## Exaone4Model

[[autodoc]] Exaone4Model
    - forward

## Exaone4ForCausalLM

[[autodoc]] Exaone4ForCausalLM
    - forward

## Exaone4ForSequenceClassification

[[autodoc]] Exaone4ForSequenceClassification
    - forward

## Exaone4ForTokenClassification

[[autodoc]] Exaone4ForTokenClassification
    - forward

## Exaone4ForQuestionAnswering

[[autodoc]] Exaone4ForQuestionAnswering
    - forward