<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-03-11 and added to Hugging Face Transformers on 2021-06-30.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# CANINE

[CANINE](https://huggingface.co/papers/2103.06874) is a tokenization-free Transformer. It skips the usual step of splitting text into subwords or wordpieces and processes text character by character. That means it works directly with raw Unicode, making it especially useful for languages with complex or inconsistent tokenization rules and even noisy inputs like typos. Since working with characters means handling longer sequences, CANINE uses a smart trick. The model compresses the input early on (called downsampling) so the transformer doesn’t have to process every character individually. This keeps things fast and efficient.

You can find all the original CANINE checkpoints under the [Google](https://huggingface.co/google?search_models=canine) organization.

> [!TIP]
> Click on the CANINE models in the right sidebar for more examples of how to apply CANINE to different language tasks.

The example below demonstrates how to generate embeddings with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="feature-extraction",
    model="google/canine-c",
    device=0,               
)

pipeline("Plant create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("google/canine-c")

text = "Plant create energy through a process known as photosynthesis."
input_ids = torch.tensor([[ord(char) for char in text]])

outputs = model(input_ids)  
pooled_output = outputs.pooler_output
sequence_output = outputs.last_hidden_state
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plant create energy through a process known as photosynthesis." | transformers-cli run --task feature-extraction --model google/canine-c --device 0
```

</hfoption>
</hfoptions>

## Notes

- CANINE skips tokenization entirely — it works directly on raw characters, not subwords. You can use it with or without a tokenizer. For batched inference and training, it is recommended to use the tokenizer to pad and truncate all sequences to the same length.

    ```py
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer("google/canine-c")
    inputs = ["Life is like a box of chocolates.", "You never know what you gonna get."]
    encoding = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")
    ```
- CANINE is primarily designed to be fine-tuned on a downstream task. The pretrained model can be used for either masked language modeling or next sentence prediction.

## CanineConfig

[[autodoc]] CanineConfig

## CanineTokenizer

[[autodoc]] CanineTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences

## CANINE specific outputs

[[autodoc]] models.canine.modeling_canine.CanineModelOutputWithPooling

## CanineModel

[[autodoc]] CanineModel
    - forward

## CanineForSequenceClassification

[[autodoc]] CanineForSequenceClassification
    - forward

## CanineForMultipleChoice

[[autodoc]] CanineForMultipleChoice
    - forward

## CanineForTokenClassification

[[autodoc]] CanineForTokenClassification
    - forward

## CanineForQuestionAnswering

[[autodoc]] CanineForQuestionAnswering
    - forward
