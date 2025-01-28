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

# Doge


## Overview

Doge is a series of small language models based on the [Doge](https://github.com/LoserCheems/WonderfulMatrices) architecture, aiming to combine the advantages of state-space and self-attention algorithms, calculate dynamic masks from cached value states using the zero-order hold method, and solve the problem of existing mainstream language models getting lost in context. It uses the `wsd_scheduler` scheduler to pre-train on the `smollm-corpus`, and can continue training on new datasets or add sparse activation feedforward networks from stable stage checkpoints.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F426/transformers/model_doc/doge_architecture.png" alt="drawing" width="600"/>

As shown in the figure below, the sequence transformation part of the Doge architecture uses `Dynamic Mask Attention`, which can be understood as using self-attention related to value states during training, and using state-space without past state decay during inference, to solve the problem of existing Transformers or SSMs getting lost in long text. The state transformation part of Doge uses `Cross Domain Mixture of Experts`, which consists of dense linear layers and sparse embedding layers, and can additionally increase sparse parameters to continue training from dense weight checkpoints without retraining the entire model, thereby reducing the cost of continuous iteration of the model. In addition, Doge also uses `RMSNorm` and `Residual` with learnable parameters to adapt the gradient range of deep models.

Checkout all Doge model checkpoints [here](https://huggingface.co/collections/JingzeShi/doge-slm-677fd879f8c4fd0f43e05458).


## Usage

<details>
<summary>Using Doge-Base for text generation</summary>

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("JingzeShi/Doge-20M")
model = AutoModelForCausalLM.from_pretrained("JingzeShi/Doge-20M")
inputs = tokenizer("Hey how are you doing?", return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.batch_decode(outputs))
```
</details>

<details>
<summary>Using Doge-Instruct for question answering</summary>

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("JingzeShi/Doge-20M-Instruct")
model = AutoModelForCausalLM.from_pretrained("JingzeShi/Doge-20M-Instruct")

generation_config = GenerationConfig(
      max_new_tokens=100, 
      use_cache=True, 
      do_sample=True, 
      temperature=0.8, 
      top_p=0.9,
      repetition_penalty=1.0
)
steamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

prompt = "Hi, how are you doing today?"
conversation = [
      {"role": "user", "content": prompt}
]
inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    tokenize=True,
    return_tensors="pt",
)

outputs = model.generate(
    inputs, 
    tokenizer=tokenizer,
    generation_config=generation_config, 
    streamer=steamer
)
```
</details>

## DogeConfig

[[autodoc]] DogeConfig

## DogeModel

[[autodoc]] DogeModel
    - forward

## DogeForCausalLM

[[autodoc]] DogeForCausalLM
    - forward

## DogeForSequenceClassification

[[autodoc]] DogeForSequenceClassification
    - forward