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
            <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
            <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# BioGPT

[BioGPT](https://huggingface.co/papers/2210.10341) is a generative Transformer model based on [GPT-2](./gpt2) and pretrained on 15 million PubMed abstracts. It is designed for biomedical language tasks.

You can find all the original BioGPT checkpoints under the [Microsoft](https://huggingface.co/microsoft?search_models=biogpt) organization.

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
    dtype=torch.float16,
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
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

input_text = "Ibuprofen is best used for"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=50)
    
output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output)
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Ibuprofen is best used for" | transformers-cli run --task text-generation --model microsoft/biogpt --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bit precision.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/BioGPT-Large", 
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto"
)

input_text = "Ibuprofen is best used for"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=50)    
output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output)
```

## Notes

- Pad inputs on the right because BioGPT uses absolute position embeddings.
- BioGPT can reuse previously computed key-value attention pairs. Access this feature with the [past_key_values](https://huggingface.co/docs/transformers/main/en/model_doc/biogpt#transformers.BioGptModel.forward.past_key_values) parameter in [`BioGPTModel.forward`].
- The `head_mask` argument is ignored when using an attention implementation other than "eager". If you want to use `head_mask`, make sure `attn_implementation="eager"`).

   ```py
   from transformers import AutoModelForCausalLM
   
   model = AutoModelForCausalLM.from_pretrained(
      "microsoft/biogpt",
      attn_implementation="eager"
   )

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