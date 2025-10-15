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
*This model was released on 2019-10-29 and added to Hugging Face Transformers on 2020-11-16 and contributed by [lysandre](https://huggingface.co/lysandre).*

# ELECTRA

[ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://huggingface.co/papers/1910.13461) proposes a novel pretraining method that uses two transformer models: a generator and a discriminator. The generator replaces tokens in a sequence with plausible alternatives, while the discriminator identifies which tokens were replaced. This approach, called replaced token detection, is more sample-efficient than masked language modeling (MLM) because it operates on all input tokens. Experiments show that ELECTRA outperforms BERT with the same resources, especially for smaller models, and performs comparably to RoBERTa and XLNet with less compute.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("fill-mask", model="google/electra-small-generator", dtype="auto")
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("google/electra-small-generator", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/electra-small-generator")

inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## Usage tips

- ELECTRA has two transformer models: a generator (G) and a discriminator (D). Use the discriminator model (indicated by `*-discriminator` in the name) for most downstream tasks.
- ELECTRA can use a smaller embedding size than the hidden size for efficiency. When `embedding_size` is smaller than `hidden_size`, a projection layer connects them.
- Use attention masks with batched inputs that have padding. This prevents the model from attending to padding tokens.
- Load the discriminator into any ELECTRA model class (`ElectraForSequenceClassification`, `ElectraForTokenClassification`, etc.) for downstream tasks.

## ElectraConfig

[[autodoc]] ElectraConfig

## ElectraTokenizer

[[autodoc]] ElectraTokenizer

## ElectraTokenizerFast

[[autodoc]] ElectraTokenizerFast

## Electra specific outputs

[[autodoc]] models.electra.modeling_electra.ElectraForPreTrainingOutput

## ElectraModel

[[autodoc]] ElectraModel
    - forward

## ElectraForPreTraining

[[autodoc]] ElectraForPreTraining
    - forward

## ElectraForCausalLM

[[autodoc]] ElectraForCausalLM
    - forward

## ElectraForMaskedLM

[[autodoc]] ElectraForMaskedLM
    - forward

## ElectraForSequenceClassification

[[autodoc]] ElectraForSequenceClassification
    - forward

## ElectraForMultipleChoice

[[autodoc]] ElectraForMultipleChoice
    - forward

## ElectraForTokenClassification

[[autodoc]] ElectraForTokenClassification
    - forward

## ElectraForQuestionAnswering

[[autodoc]] ElectraForQuestionAnswering
    - forward

