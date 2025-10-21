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
*This model was released on 2020-06-05 and added to Hugging Face Transformers on 2020-11-16 and contributed by [sgugger](https://huggingface.co/sgugger).*

# Funnel Transformer

[Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing](https://huggingface.co/papers/2006.03236) proposes a bidirectional transformer model that incorporates pooling operations after each block of layers, similar to CNNs. This model compresses the sequence of hidden states to reduce computational cost, allowing for deeper or wider architectures. It can also recover deep representations for each token via a decoder, enabling token-level predictions. Empirically, Funnel-Transformer outperforms standard Transformers on various sequence-level prediction tasks with comparable or fewer floating-point operations.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="funnel-transformer/small", dtype="auto")
pipeline("Plants create <mask> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("funnel-transformer/small", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/small")

inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## Usage tips

- Funnel Transformer uses pooling, so sequence length changes after each block. Length divides by 2, speeding up computation. The base model has a final sequence length that's a quarter of the original.
- Use the base model directly for tasks requiring sentence summaries (sequence classification or multiple choice). Use the full model for other tasks. The full model has a decoder that upsamples final hidden states to match input sequence length.
- For classification tasks, this works fine. For masked language modeling or token classification, you need hidden states with the same sequence length as the original input. Final hidden states get upsampled to input sequence length and go through two additional layers.
- Two checkpoint versions exist. The `-base` version contains only three blocks. The version without that suffix contains three blocks plus the upsampling head with additional layers.

## FunnelConfig

[[autodoc]] FunnelConfig

## FunnelTokenizer

[[autodoc]] FunnelTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FunnelTokenizerFast

[[autodoc]] FunnelTokenizerFast

## Funnel specific outputs

[[autodoc]] models.funnel.modeling_funnel.FunnelForPreTrainingOutput

## FunnelBaseModel

[[autodoc]] FunnelBaseModel
    - forward

## FunnelModel

[[autodoc]] FunnelModel
    - forward

## FunnelModelForPreTraining

[[autodoc]] FunnelForPreTraining
    - forward

## FunnelForMaskedLM

[[autodoc]] FunnelForMaskedLM
    - forward

## FunnelForSequenceClassification

[[autodoc]] FunnelForSequenceClassification
    - forward

## FunnelForMultipleChoice

[[autodoc]] FunnelForMultipleChoice
    - forward

## FunnelForTokenClassification

[[autodoc]] FunnelForTokenClassification
    - forward

## FunnelForQuestionAnswering

[[autodoc]] FunnelForQuestionAnswering
    - forward

