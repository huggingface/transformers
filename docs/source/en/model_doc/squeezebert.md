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
*This model was released on 2020-06-19 and added to Hugging Face Transformers on 2020-11-16 and contributed by [forresti](https://huggingface.co/forresti).*

# SqueezeBERT

[SqueezeBERT](https://huggingface.co/papers/2006.11316) is a bidirectional transformer model that leverages grouped convolutions in place of fully-connected layers for Q, K, V, and FFN layers, significantly enhancing efficiency. This approach enables SqueezeBERT to run 4.3x faster than BERT-base on a Pixel 3 smartphone while maintaining competitive accuracy on the GLUE test set.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="squeezebert/squeezebert-uncased", dtype="auto")
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("squeezebert/squeezebert-uncased", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")

inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## SqueezeBertConfig

[[autodoc]] SqueezeBertConfig

## SqueezeBertTokenizer

[[autodoc]] SqueezeBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## SqueezeBertTokenizerFast

[[autodoc]] SqueezeBertTokenizerFast

## SqueezeBertModel

[[autodoc]] SqueezeBertModel

## SqueezeBertForMaskedLM

[[autodoc]] SqueezeBertForMaskedLM

## SqueezeBertForSequenceClassification

[[autodoc]] SqueezeBertForSequenceClassification

## SqueezeBertForMultipleChoice

[[autodoc]] SqueezeBertForMultipleChoice

## SqueezeBertForTokenClassification

[[autodoc]] SqueezeBertForTokenClassification

## SqueezeBertForQuestionAnswering

[[autodoc]] SqueezeBertForQuestionAnswering

