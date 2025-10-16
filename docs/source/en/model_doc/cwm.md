<!-- Copyright 2025 the HuggingFace Team. All rights reserved.

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
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-10-09.*

# Code World Model (CWM)

[CWM](https://huggingface.co/papers/2510.02387) is a 32-billion-parameter open-weight language model designed to improve code generation through world modeling. It is mid-trained on extensive observation-action data from Python and Docker environments, then refined with reinforcement learning across coding, math, and software engineering tasks. The model supports step-by-step simulation of code execution and demonstrates strong reasoning and planning abilities. Technically, CWM is a dense, decoder-only model with a 131k-token context window, achieving state-of-the-art results on SWE-bench Verified (65.8%), LiveCodeBench (68.6%), Math-500 (96.6%), and AIME 2024 (76.0%).

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="facebook/cwm", dtype="auto")
pipeline("Plants generate energy through a process known as  ")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("facebook/cwm", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/cwm")

system_prompt = """
You are a helpful AI assistant. You always reason before responding, using the following format:

<think>
your internal reasoning
</think>
your external response
""".strip()

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Write a haiku about recursion in programming."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
    preserve_previous_think=True
)

model_inputs = tokenizer([text], return_tensors="pt")

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
print(tokenizer.decode(output_ids))
```

</hfoption>
</hfoptions>

## CwmConfig

[[autodoc]] CwmConfig

## CwmPreTrainedModel

[[autodoc]] CwmPreTrainedModel
    - forward

## CwmModel

[[autodoc]] CwmModel
    - forward

## CwmForCausalLM

[[autodoc]] CwmForCausalLM
