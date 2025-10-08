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
*This model was released on 2020-10-24 and added to Hugging Face Transformers on 2021-07-24.*

# RemBERT

[RemBERT](https://huggingface.co/papers/2010.12821) re-evaluates the practice of sharing weights between input and output embeddings in pre-trained language models. It demonstrates that decoupling these embeddings enhances modeling flexibility and optimizes parameter allocation in multilingual models. By reallocating input embedding parameters within Transformer layers, the model achieves superior performance on natural language understanding tasks without increasing parameters during fine-tuning. Additionally, allocating extra capacity to the output embedding, which is discarded post-pre-training, improves the model's generalization and transferability to various tasks and languages. These insights enable strong performance on the XTREME benchmark without expanding the parameter count at the fine-tuning stage.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="google/rembert", dtype="auto")
pipeline("Plants create <mask> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("google/rembert", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/rembert")

inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## RemBertConfig

[[autodoc]] RemBertConfig

## RemBertTokenizer

[[autodoc]] RemBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RemBertTokenizerFast

[[autodoc]] RemBertTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RemBertModel

[[autodoc]] RemBertModel
    - forward

## RemBertForCausalLM

[[autodoc]] RemBertForCausalLM
    - forward

## RemBertForMaskedLM

[[autodoc]] RemBertForMaskedLM
    - forward

## RemBertForSequenceClassification

[[autodoc]] RemBertForSequenceClassification
    - forward

## RemBertForMultipleChoice

[[autodoc]] RemBertForMultipleChoice
    - forward

## RemBertForTokenClassification

[[autodoc]] RemBertForTokenClassification
    - forward

## RemBertForQuestionAnswering

[[autodoc]] RemBertForQuestionAnswering
    - forward

