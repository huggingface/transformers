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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
            <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
            <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# BioGPT

[BioGPT](https://huggingface.co/papers/2210.10341) is a generative Transformer model based on the GPT-2 architecture and pre-trained on 15 million PubMed abstracts. That makes it great at writing, understanding, and analyzing biomedical text.

You can find all the original BioGPT checkpoints under the [BioGPT](https://huggingface.co/models?search=biogpt) collection.

> [!TIP]
> Click on the BioGPT models in the right sidebar for more examples of how to apply BioGPT to different language tasks.

The example below demonstrates how to generate biomedical text with [`Pipeline`], [`AutoModel`], and also from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

generator = pipeline(
    task="text-generation",
    model="microsoft/biogpt",
    torch_dtype=torch.float16,
    device=0,
)
result = generator("Ibuprofen is best used for", truncation=True, max_length=50, do_sample=True)[0]["generated_text"]
print(result)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/biogpt",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

input_text = "Ibuprofen is best used for"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=50)
    
output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output)
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Ibuprofen is best used for" | transformers run --task text-generation --model microsoft/biogpt --device 0
```

</hfoption>
</hfoptions>

## Notes

- BioGPT is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than the left.
- BioGPT was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence. Leveraging this feature allows BioGPT to generate syntactically coherent text as it can be observed in the run_generation.py example script.
- The model can take the `past_key_values` (for PyTorch) as input, which is the previously computed key/value attention pairs. Using this (past_key_values or past) value prevents the model from re-computing pre-computed values in the context of text generation. For PyTorch, see past_key_values argument of the BioGptForCausalLM.forward() method for more information on its usage.
- The `head_mask` argument is ignored when using all attention implementation other than "eager". If you have a `head_mask` and want it to have effect, load the model with `XXXModel.from_pretrained(model_id, attn_implementation="eager")`

## BioGptConfig

[[autodoc]] BioGptConfig


## BioGptTokenizer

[[autodoc]] BioGptTokenizer
    - save_vocabulary


## BioGptModel

[[autodoc]] BioGptModel
    - forward


## BioGptForCausalLM

[[autodoc]] BioGptForCausalLM
    - forward

    
## BioGptForTokenClassification

[[autodoc]] BioGptForTokenClassification
    - forward


## BioGptForSequenceClassification

[[autodoc]] BioGptForSequenceClassification
    - forward