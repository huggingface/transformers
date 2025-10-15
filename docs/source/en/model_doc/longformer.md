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
*This model was released on 2020-04-10 and added to Hugging Face Transformers on 2020-11-16 and contributed by [beltagy](https://huggingface.co/beltagy).*

# Longformer

[Longformer: The Long-Document Transformer](https://huggingface.co/papers/2004.05150) introduces an attention mechanism that scales linearly with sequence length, enabling the processing of very long documents. This mechanism combines local windowed attention with task-specific global attention, replacing standard self-attention. Longformer achieves state-of-the-art results in character-level language modeling on text8 and enwik8. When pretrained and fine-tuned, it outperforms RoBERTa on long document tasks, setting new benchmarks on WikiHop and TriviaQA.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="allenai/longformer-base-4096", dtype="auto")
pipeline("Plants are among the most <mask> and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet. Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems. These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("allenai/longformer-base-4096", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

text="""
Plants are among the most <mask> and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
"""
input_ids = tokenizer([text], return_tensors="pt")["input_ids"]
logits = model(input_ids).logits
print(tokenizer.decode(logits[0, (input_ids[0] == tokenizer.mask_token_id).nonzero().item()].argmax()))
```

</hfoption>
</hfoptions>

## Usage tips

- Longformer is based on RoBERTa and doesn't have `token_type_ids`. You don't need to indicate which token belongs to which segment. Just separate segments with the separation token `</s>` or `tokenizer.sep_token`.
- Set which tokens attend locally and which attend globally with the `global_attention_mask` at inference. A value of 0 means a token attends locally. A value of 1 means a token attends globally.
- [`LongformerForMaskedLM`] is trained like [`RobertaForMaskedLM`] and should be similarly.

## LongformerConfig

[[autodoc]] LongformerConfig

## LongformerTokenizer

[[autodoc]] LongformerTokenizer

## LongformerTokenizerFast

[[autodoc]] LongformerTokenizerFast

## Longformer specific outputs

[[autodoc]] models.longformer.modeling_longformer.LongformerBaseModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerBaseModelOutputWithPooling

[[autodoc]] models.longformer.modeling_longformer.LongformerMaskedLMOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerSequenceClassifierOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerMultipleChoiceModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerTokenClassifierOutput

] models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutputWithPooling

] models.longformer.modeling_tf_longformer.TFLongformerQuestionAnsweringModelOutput

] models.longformer.modeling_tf_longformer.TFLongformerMultipleChoiceModelOutput

## LongformerModel

[[autodoc]] LongformerModel
    - forward

## LongformerForMaskedLM

[[autodoc]] LongformerForMaskedLM
    - forward

## LongformerForSequenceClassification

[[autodoc]] LongformerForSequenceClassification
    - forward

## LongformerForMultipleChoice

[[autodoc]] LongformerForMultipleChoice
    - forward

## LongformerForTokenClassification

[[autodoc]] LongformerForTokenClassification
    - forward

## LongformerForQuestionAnswering

[[autodoc]] LongformerForQuestionAnswering
    - forward

