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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# CANINE

[CANINE](https://huggingface.co/papers/2103.06874) is a tokenization-free Transformer, It skips the usual step of splitting text into subwords or wordpieces and processes text character by character. That means it works directly with raw Unicode, making it especially useful for languages with complex or inconsistent tokenization rules and even noisy inputs like typos. Since working with characters means handling longer sequences, CANINE uses a smart trick. It compresses the input early on (called downsampling) so the transformer doesn’t have to process every character individually. This keeps things fast and efficient.

You can find all the original CANINE checkpoints under the [Google](https://huggingface.co/google?search_models=canine) organization.

> [!TIP]
> Click on the CANINE models in the right sidebar for more examples of how to apply CANINE to different language tasks.

The example below demonstrates how to generate embeddings with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

extractor = pipeline(
    task="feature-extraction",
    model="google/canine-c",
    tokenizer="google/canine-c",
    device=0,               
)

text = "Hello World!"
embeddings = extractor(text)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("google/canine-c")

text = "Hello World!"
input_ids = torch.tensor([[ord(char) for char in text]])

outputs = model(input_ids)  
pooled_output = outputs.pooler_output
sequence_output = outputs.last_hidden_state
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Hello World!" | transformers-cli run \
  --task feature-extraction \
  --model google/canine-c \
  --device 0
```

</hfoption>
</hfoptions>

## Notes

- CANINE skips tokenization entirely — it works directly on raw characters, not subwords. You can use it with or without a tokenizer (e.g., CanineTokenizer is available if needed for consistency or padding).
- The model supports sequences up to 2048 characters by default.
- Classification can be done by placing a linear layer on top of the final hidden state of the special [CLS] token
  (which has a predefined Unicode code point). For token classification tasks however, the downsampled sequence of
  tokens needs to be upsampled again to match the length of the original character sequence (which is 2048). The
  details for this can be found in the paper.

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
