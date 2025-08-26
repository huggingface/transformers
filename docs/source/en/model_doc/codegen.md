---

# CodeGen

[![](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/) ![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

[CodeGen: A Conversational Paradigm for Program Synthesis](https://huggingface.co/papers/2203.13474)

CodeGen is an autoregressive language model for code generation and conversational program synthesis, released by Salesforce in 2022. It is trained on natural language and programming language data, scaling to 16B parameters, and is designed to synthesize code in response to multi-turn, conversational prompts. Notably, CodeGen outperforms OpenAI Codex on the HumanEval benchmark. For more details, see the [training library and checkpoints](https://github.com/salesforce/codegen).

> **TIP**: CodeGen models are contributed and maintained by the Salesforce research team.

## Model Usage Examples

### Using with pipeline
```python
from transformers import pipeline
# Text-to-code pipeline (e.g. codegen-2B-mono)
generator = pipeline("text-generation", model="Salesforce/codegen-2B-mono")
output = generator("# Python function to compute factorial\ndef factorial(n):", max_new_tokens=32)
print(output[0]["generated_text"])
```

### Using AutoModelForCausalLM
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

checkpoint = "Salesforce/codegen-2B-mono"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
inputs = tokenizer("def fibonacci(n):", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Quantization
If you want to speed up inference and/or reduce memory usage, you can quantize CodeGen with supported backends. See the [Quantization Guide](../quantization/overview).

### Attention Visualization
To visualize which tokens CodeGen attends to, use:
```python
from transformers.utils.attention_visualizer import AttentionMaskVisualizer
visualizer = AttentionMaskVisualizer("Salesforce/codegen-2B-mono")
visualizer("# Python script to merge two sorted lists\ndef merge(l1, l2):")
```

## Notes
- CodeGen is intended primarily for code generation. For conversational turn-taking, ensure you prompt with explicit docstrings and context for best results.
- Full model checkpoints and training scripts are at [https://github.com/salesforce/codegen](https://github.com/salesforce/codegen).

## Resources
- [CodeGen research paper (arXiv)](https://arxiv.org/abs/2203.13474)
- [HumanEval benchmark](https://github.com/openai/human-eval)
- [Salesforce CodeGen Hub page](https://huggingface.co/Salesforce/codegen-2B-mono)
---

<!--Copyright 2022 The HuggingFace Team. All rights reserved.Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->
*
This model was released on 2022-03-25 and added to Hugging Face Transformers on 2022-06-24.
*
#
 CodeGen<div>class
=
"flex flex-wrap space-x-1"
&gt;alt
=
"PyTorch"
 
src
=
"https://img.shields.io/badge/PyTorch-DE3412?style=flat
&amp;
logo=pytorch
&amp;
logoColor=white"
&gt;</div>&gt;##
 Overview
The CodeGen model was proposed in 
[
A Conversational Paradigm for Program Synthesis
]
(
https://huggingface.co/papers/2203.13474
)
 by Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong.
CodeGen is an autoregressive language model for program synthesis trained sequentially on 
[
The Pile
]
(
https://pile.eleuther.ai/
)
, BigQuery, and BigPython.
The abstract from the paper is the following:
*
Program synthesis strives to generate a computer program as a solution to a given problem specification. We propose a conversational program synthesis approach via large language models, which addresses the challenges of searching over a vast program space and user intent specification faced in prior approaches. Our new approach casts the process of writing a specification and program as a multi-turn conversation between a user and a system. It treats program synthesis as a sequence prediction problem, in w
hich the specification is expressed in natural language and the desired program is conditionally sampled. We train a family of large language models, called CodeGen, on natural language and programming language data. With weak supervision in the data and the scaling up of data size and model size, conversational capacities emerge from the simple autoregressive language modeling. To study the model behavior on conversational program synthesis, we develop a multi-turn programming benchmark (MTPB), where solvi
ng each problem requires multi-step synthesis via multi-turn conversation between the user and the model. Our findings show the emergence of conversational capabilities and the effectiveness of the proposed conversational program synthesis paradigm. In addition, our model CodeGen (with up to 16B parameters trained on TPU-v4) outperforms OpenAI's Codex on the HumanEval benchmark. We make the training library JaxFormer including checkpoints available as open source contribution: 
[
this https URL
]
(
https://github.com/salesforce/codegen
)
.
*
 
