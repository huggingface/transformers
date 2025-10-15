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
*This model was released on 2020-04-06 and added to Hugging Face Transformers on 2020-11-16 and contributed by [vshampor](https://huggingface.co/vshampor).*

# MobileBERT

[MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://huggingface.co/papers/2004.02984) is a bidirectional transformer model designed to compress and accelerate BERT for mobile devices. It maintains task-agnostic applicability through simple fine-tuning. MobileBERT uses bottleneck structures and balances self-attentions with feed-forward networks. Trained via knowledge transfer from an inverted-bottleneck BERT_LARGE teacher model, MobileBERT is 4.3x smaller and 5.5x faster than BERT_BASE. It achieves competitive results on GLUE with a GLUEscore of 77.7 and 62 ms latency on a Pixel 4 phone, and on SQuAD v1.1/v2.0 with dev F1 scores of 90.0/79.2.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="google/mobilebert-uncased", dtype="auto")
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("google/mobilebert-uncased", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

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

- Pad inputs on the right. MobileBERT uses absolute position embeddings.

## MobileBertConfig

[[autodoc]] MobileBertConfig

## MobileBertTokenizer

[[autodoc]] MobileBertTokenizer

## MobileBertTokenizerFast

[[autodoc]] MobileBertTokenizerFast

## MobileBert specific outputs

[[autodoc]] models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput

## MobileBertModel

[[autodoc]] MobileBertModel
    - forward

## MobileBertForPreTraining

[[autodoc]] MobileBertForPreTraining
    - forward

## MobileBertForMaskedLM

[[autodoc]] MobileBertForMaskedLM
    - forward

## MobileBertForNextSentencePrediction

[[autodoc]] MobileBertForNextSentencePrediction
    - forward

## MobileBertForSequenceClassification

[[autodoc]] MobileBertForSequenceClassification
    - forward

## MobileBertForMultipleChoice

[[autodoc]] MobileBertForMultipleChoice
    - forward

## MobileBertForTokenClassification

[[autodoc]] MobileBertForTokenClassification
    - forward

## MobileBertForQuestionAnswering

[[autodoc]] MobileBertForQuestionAnswering
    - forward

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="google/mobilebert-uncased", dtype="auto")
pipeline("The capital of France is [MASK].")
```

