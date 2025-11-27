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

# NanoChat

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

[NanoChat](https://huggingface.co/karpathy/nanochat-d32) is a compact decoder-only transformer model designed for educational purposes and efficient training. The model features several fundamental architectural innovations which are common in modern transformer models. Therefore, it is a good model to use as a starting point to understand the principles of modern transformer models. NanoChat is a variant of the [Llama](https://huggingface.co/docs/transformers/en/model_doc/llama) architecture, with simplified attention mechanism and normalization layers. 

The architecture is based on [nanochat](https://github.com/karpathy/nanochat) by [Andrej Karpathy](https://huggingface.co/karpathy), adapted for the Hugging Face Transformers library by [Ben Burtenshaw](https://huggingface.co/burtenshaw).

> [!TIP]
> This model was contributed by the Hugging Face team.

The example below demonstrates how to use NanoChat for text generation with chat templates.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

chatbot = pipeline(
    task="text-generation",
    model="karpathy/nanochat-d32",
    dtype=torch.bfloat16,
    device=0
)

conversation = [
    {"role": "user", "content": "What is the capital of France?"},
]

outputs = chatbot(conversation, max_new_tokens=64)
print(outputs[0]["generated_text"][-1]["content"])
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "karpathy/nanochat-d32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

conversation = [
    {"role": "user", "content": "What is the capital of France?"},
]

inputs = tokenizer.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
    )

# Decode only the generated tokens (excluding the input prompt)
generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e '{"role": "user", "content": "What is the capital of France?"}' | transformers run --task text-generation --model karpathy/nanochat-d32 --device 0
```

</hfoption>
</hfoptions>

## NanoChatConfig

[[autodoc]] NanoChatConfig

## NanoChatModel

[[autodoc]] NanoChatModel
    - forward

## NanoChatForCausalLM

[[autodoc]] NanoChatForCausalLM
    - forward
