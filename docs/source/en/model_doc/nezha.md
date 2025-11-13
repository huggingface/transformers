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
*This model was released on 2019-08-31 and added to Hugging Face Transformers on 2023-06-20 and contributed by [sijunhe](https://huggingface.co/sijunhe).*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code.
>
> If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2. You can do so by running the following command: pip install -U transformers==4.40.2.

# Nezha

[NEZHA: Neural ContextualiZed representation for CHinese lAnguage understanding](https://huggingface.co/papers/1909.00204) presents NEZHA, a pre-trained language model for Chinese NLU tasks. NEZHA is based on BERT with enhancements such as Functional Relative Positional Encoding, Whole Word Masking, Mixed Precision Training, and the LAMB Optimizer. Experiments demonstrate that NEZHA achieves top performance on tasks like named entity recognition, sentence matching, sentiment classification, and natural language inference.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="sijunhe/nezha-cn-base", dtype="auto")
pipeline("植物通过[MASK]合作用产生能量")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("sijunhe/nezha-cn-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("sijunhe/nezha-cn-base")

inputs = tokenizer("植物通过[MASK]合作用产生能量", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## NezhaConfig

[[autodoc]] NezhaConfig

## NezhaModel

[[autodoc]] NezhaModel
    - forward

## NezhaForPreTraining

[[autodoc]] NezhaForPreTraining
    - forward

## NezhaForMaskedLM

[[autodoc]] NezhaForMaskedLM
    - forward

## NezhaForNextSentencePrediction

[[autodoc]] NezhaForNextSentencePrediction
    - forward

## NezhaForSequenceClassification

[[autodoc]] NezhaForSequenceClassification
    - forward

## NezhaForMultipleChoice

[[autodoc]] NezhaForMultipleChoice
    - forward

## NezhaForTokenClassification

[[autodoc]] NezhaForTokenClassification
    - forward

## NezhaForQuestionAnswering

[[autodoc]] NezhaForQuestionAnswering
    - forward

