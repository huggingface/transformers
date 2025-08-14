<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    <img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,...">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
  </div>
</div>



# GPT

[GPT (Generative Pre-trained Transformer)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) focuses on effectively learning text representations and transferring them to tasks. This model trains the Transformer decoder to predict the next word, and then fine-tuned on labeled data.

GPT can generate high-quality text, making it well-suited for a variety of natural language understanding tasks such as textual entailment, question answering, semantic similarity, and document classification.

You can find all the original GPT checkpoints under the [OpenAI community](https://huggingface.co/openai-community/openai-gpt) organization.

> [!TIP]
> Click on the GPT models in the right sidebar for more examples of how to apply GPT to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.



<hfoptions id="usage">
<hfoption id="Pipeline">


```python
import torch
from transformers import pipeline

generator = pipeline(task="text-generation", model="openai-community/gpt", dtype=torch.float16, device=0)
output = generator("The future of AI is", max_length=50, do_sample=True)
print(output[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt")
model = AutoModelForCausalLM.from_pretrained("openai-community/openai-gpt", dtype=torch.float16)

inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "The future of AI is" | transformers run --task text-generation --model openai-community/openai-gpt --device 0

```
</hfoption>
</hfoptions>

## Notes

- Inputs should be padded on the right because GPT uses absolute position embeddings.

## OpenAIGPTConfig

[[autodoc]] OpenAIGPTConfig

## OpenAIGPTModel

[[autodoc]] OpenAIGPTModel
- forward

## OpenAIGPTLMHeadModel

[[autodoc]] OpenAIGPTLMHeadModel
- forward

## OpenAIGPTDoubleHeadsModel

[[autodoc]] OpenAIGPTDoubleHeadsModel
- forward

## OpenAIGPTForSequenceClassification

[[autodoc]] OpenAIGPTForSequenceClassification
- forward

## OpenAIGPTTokenizer

[[autodoc]] OpenAIGPTTokenizer

## OpenAIGPTTokenizerFast

[[autodoc]] OpenAIGPTTokenizerFast

## TFOpenAIGPTModel

[[autodoc]] TFOpenAIGPTModel
- call

## TFOpenAIGPTLMHeadModel

[[autodoc]] TFOpenAIGPTLMHeadModel
- call

## TFOpenAIGPTDoubleHeadsModel

[[autodoc]] TFOpenAIGPTDoubleHeadsModel
- call

## TFOpenAIGPTForSequenceClassification

[[autodoc]] TFOpenAIGPTForSequenceClassification
- call
