<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-03-25 and added to Hugging Face Transformers on 2022-06-24 and contributed by [rooa](https://huggingface.co/rooa).*

# CodeGen

[CodeGen](https://huggingface.co/papers/2203.13474) is an autoregressive language model designed for program synthesis through a conversational paradigm. Trained on diverse datasets including The Pile, BigQuery, and BigPython, CodeGen addresses challenges in program synthesis by treating it as a sequence prediction problem where specifications are expressed in natural language. The model demonstrates conversational capabilities and outperforms OpenAI's Codex on the HumanEval benchmark. A multi-turn programming benchmark (MTPB) was developed to evaluate the model's conversational program synthesis abilities. 

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Salesforce/codegen-350M-mono", dtype="auto")
pipeline("def fibonacci(n):")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

inputs = tokenizer("def fibonacci(n):", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## CodeGenConfig

[[autodoc]] CodeGenConfig
    - all

## CodeGenTokenizer

[[autodoc]] CodeGenTokenizer
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CodeGenTokenizerFast

[[autodoc]] CodeGenTokenizerFast

## CodeGenModel

[[autodoc]] CodeGenModel
    - forward

## CodeGenForCausalLM

[[autodoc]] CodeGenForCausalLM
    - forward

