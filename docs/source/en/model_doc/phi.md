<!--Copyright 2023 The HuggingFace Team. All rights reserved.

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
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Phi

[Phi](https://huggingface.co/papers/2306.11644) is a 1.3B parameter transformer model optimized for Python code generation. It focuses on "textbook-quality" training data of code examples, exercises and synthetic Python problems rather than scaling the model size or compute.

You can find all the original Phi checkpoints under the [Phi-1](https://huggingface.co/collections/microsoft/phi-1-6626e29134744e94e222d572) collection.

> [!TIP]
> Click on the Phi models in the right sidebar for more examples of how to apply Phi to different language tasks.

The example below demonstrates how to generate and classify text with [`Pipeline`], [`AutoModel`] and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
task="text-generation",
model="microsoft/phi-1.5"
)
message = "Why Lebanon is a special country?"
output = pipe(message)

```

</hfoption>

<hfoption id="AutoModel">

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)

tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

</hfoption>
<hfoption id="transformers-cli">

```bash
echo -e "The weather is so nice here" | transformers-cli run --task text-classification --model microsoft/phi-1.5 --device 0
```

</hfoption>
</hfoptions>

## Notes

- This model is quite similar to `Llama` with the main difference in `PhiDecoderLayer`, where they used `PhiAttention` and `PhiMLP` layers in parallel configuration.

- The tokenizer used for this model is identical to the [CodeGenTokenizer](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/codegen#transformers.CodeGenTokenizer).

 ## PhiConfig
[[autodoc]] PhiConfig

## PhiModel

[[autodoc]] PhiModel - forward

## PhiForCausalLM

[[autodoc]] PhiForCausalLM - forward - generate

## PhiForSequenceClassification

[[autodoc]] PhiForSequenceClassification - forward

## PhiForTokenClassification

[[autodoc]] PhiForTokenClassification - forward
