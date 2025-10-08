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
*This model was released on 2021-04-20 and added to Hugging Face Transformers on 2021-05-20 and contributed by [junnyu](https://huggingface.co/junnyu).*

# RoFormer

[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://huggingface.co/papers/2104.09864v1) proposes Rotary Position Embedding (RoPE) to encode positional information in transformer-based language models. RoPE uses a rotation matrix to encode absolute positions and naturally integrates relative position dependencies into self-attention. Key benefits include flexibility for varying sequence lengths, decreasing inter-token dependencies with distance, and enabling relative position encoding in linear self-attention. RoFormer demonstrates superior performance on long texts, with theoretical analysis and preliminary results on Chinese data provided.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="junnyu/roformer_chinese_base", dtype="auto")
pipeline("植物通过[MASK]合作用产生能量")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("junnyu/roformer_chinese_base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("junnyu/roformer_chinese_base")

inputs = tokenizer("植物通过[MASK]合作用产生能量", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## RoFormerConfig

[[autodoc]] RoFormerConfig

## RoFormerTokenizer

[[autodoc]] RoFormerTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RoFormerTokenizerFast

[[autodoc]] RoFormerTokenizerFast
    - build_inputs_with_special_tokens

## RoFormerModel

[[autodoc]] RoFormerModel
    - forward

## RoFormerForCausalLM

[[autodoc]] RoFormerForCausalLM
    - forward

## RoFormerForMaskedLM

[[autodoc]] RoFormerForMaskedLM
    - forward

## RoFormerForSequenceClassification

[[autodoc]] RoFormerForSequenceClassification
    - forward

## RoFormerForMultipleChoice

[[autodoc]] RoFormerForMultipleChoice
    - forward

## RoFormerForTokenClassification

[[autodoc]] RoFormerForTokenClassification
    - forward

## RoFormerForQuestionAnswering

[[autodoc]] RoFormerForQuestionAnswering
    - forward

