<!--Copyright 2024 The GLM & ZhipuAI team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GLM

## Overview

The GLM Model was proposed
in [ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools](https://arxiv.org/html/2406.12793v1)
by GLM Team, THUDM & ZhipuAI.

The abstract from the paper is the following:

*We introduce ChatGLM, an evolving family of large language models that we have been developing over time. This report
primarily focuses on the GLM-4 language series, which includes GLM-4, GLM-4-Air, and GLM-4-9B. They represent our most
capable models that are trained with all the insights and lessons gained from the preceding three generations of
ChatGLM. To date, the GLM-4 models are pre-trained on ten trillions of tokens mostly in Chinese and English, along with
a small set of corpus from 24 languages, and aligned primarily for Chinese and English usage. The high-quality alignment
is achieved via a multi-stage post-training process, which involves supervised fine-tuning and learning from human
feedback. Evaluations show that GLM-4 1) closely rivals or outperforms GPT-4 in terms of general metrics such as MMLU,
GSM8K, MATH, BBH, GPQA, and HumanEval, 2) gets close to GPT-4-Turbo in instruction following as measured by IFEval, 3)
matches GPT-4 Turbo (128K) and Claude 3 for long context tasks, and 4) outperforms GPT-4 in Chinese alignments as
measured by AlignBench. The GLM-4 All Tools model is further aligned to understand user intent and autonomously decide
when and which tool(s) to use—including web browser, Python interpreter, text-to-image model, and user-defined
functions—to effectively complete complex tasks. In practical applications, it matches and even surpasses GPT-4 All
Tools in tasks like accessing online information via web browsing and solving math problems using Python interpreter.
Over the course, we have open-sourced a series of models, including ChatGLM-6B (three generations), GLM-4-9B (128K, 1M),
GLM-4V-9B, WebGLM, and CodeGeeX, attracting over 10 million downloads on Hugging face in the year 2023 alone.*

Tips:

- This model was contributed by [THUDM](https://huggingface.co/THUDM). The most recent code can be
  found [here](https://github.com/thudm/GLM-4).

  
## Usage tips

`GLM-4` can be found on the [Huggingface Hub](https://huggingface.co/collections/THUDM/glm-4-665fcf188c414b03c2f7e3b7)

In the following, we demonstrate how to use `glm-4-9b-chat` for the inference. Note that we have used the ChatML format for dialog, in this demo we show how to leverage `apply_chat_template` for this purpose.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # the device to load the model onto

>>> model = AutoModelForCausalLM.from_pretrained("THUDM/glm-4-9b-chat", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat")

>>> prompt = "Give me a short introduction to large language model."

>>> messages = [{"role": "user", "content": prompt}]

>>> text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

>>> model_inputs = tokenizer([text], return_tensors="pt").to(device)

>>> generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

>>> generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

>>> response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## GLMConfig

[[autodoc]] GLMConfig

## GLMTokenizer

[[autodoc]] GLMTokenizer
    - save_vocabulary

## GLMTokenizerFast

[[autodoc]] GLMTokenizerFast

## GLMModel

[[autodoc]] GLMModel
    - forward

## GLMForCausalLM

[[autodoc]] GLMForCausalLM
    - forward

## GLMForSequenceClassification

[[autodoc]] GLMForSequenceClassification
    - forward

## GLMForTokenClassification

[[autodoc]] GLMForTokenClassification
    - forward
