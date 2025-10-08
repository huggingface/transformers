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

*This model was released on 2024-12-27 and added to Hugging Face Transformers on 2025-07-08.*

# Doge

[Doge-20M](https://huggingface.co/papers/PAPER_ID) is utilized for text generation, demonstrating its capability to produce coherent and contextually relevant responses. For question answering, Doge-20M-Instruct is employed, showcasing enhanced performance in understanding and generating answers through a structured conversational format. The model leverages specific generation configurations, including temperature and top-p sampling, to ensure varied and engaging outputs.

## Usage

<details>
<summary>Using Doge-Base for text generation</summary>

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-20M")
model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-20M")
inputs = tokenizer("Hey how are you doing?", return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.batch_decode(outputs))
```

</details>

<details>
<summary>Using Doge-Instruct for question answering</summary>

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-20M-Instruct")
model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-20M-Instruct")

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

