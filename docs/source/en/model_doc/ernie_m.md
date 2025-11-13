<!--Copyright 2023 The HuggingFace and Baidu Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-12-31 and added to Hugging Face Transformers on 2023-06-20 and contributed by [susnato](https://huggingface.co/susnato).*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code.
>
> If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2. You can do so by running the following command: pip install -U transformers==4.40.2.

# ErnieM

[ERNIE-M: Enhanced Multilingual Representation by Aligning Cross-lingual Semantics with Monolingual Corpora](https://huggingface.co/papers/2012.15674) proposes a new training method called ERNIE-M to improve cross-lingual model performance, especially for low-resource languages. By integrating back-translation into the pre-training process, ERNIE-M generates pseudo-parallel sentence pairs from monolingual corpora to learn semantic alignments between different languages. This approach enhances semantic modeling in cross-lingual models, leading to state-of-the-art results in various downstream tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-classification", model="susnato/ernie-m-base_pytorch", dtype="auto")
pipeline("Plants are amazing because they can create energy from the sun.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("susnato/ernie-m-base_pytorch", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("susnato/ernie-m-base_pytorch")

inputs = tokenizer("Plants are amazing because they can create energy from the sun.", return_tensors="pt")
outputs = model(**inputs)
predicted_class_id = outputs.logits.argmax(dim=-1).item()
label = model.config.id2label[predicted_class_id]
print(f"Predicted label: {label}")
```

</hfoption>
</hfoptions>

## Usage tips

- ERNIE-M uses two novel techniques instead of MaskedLM for pretraining: Cross-attention Masked Language Modeling and Back-translation Masked Language Modeling. These LMHead objectives aren't implemented yet.

## ErnieMConfig

[[autodoc]] ErnieMConfig

## ErnieMTokenizer

[[autodoc]] ErnieMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## ErnieMModel

[[autodoc]] ErnieMModel
    - forward

## ErnieMForSequenceClassification

[[autodoc]] ErnieMForSequenceClassification
    - forward

## ErnieMForMultipleChoice

[[autodoc]] ErnieMForMultipleChoice
    - forward

## ErnieMForTokenClassification

[[autodoc]] ErnieMForTokenClassification
    - forward

## ErnieMForQuestionAnswering

[[autodoc]] ErnieMForQuestionAnswering
    - forward

## ErnieMForInformationExtraction

[[autodoc]] ErnieMForInformationExtraction
    - forward

