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


<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    </div>
</div>

# MobileBERT

[MobileBERT](https://github.com/google-research/google-research/tree/master/mobilebert) is a lightweight and efficient variant of BERT, specifically designed for resource-limited devices such as mobile phones. It retains the bidirectional transformer architecture of BERT but significantly reduces model size and inference latency while maintaining strong performance on NLP tasks.
The MobileBERT model was proposed in [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984). It's a bidirectional transformer based on the BERT model, which is compressed and accelerated using several
approaches.

> [!TIP]
> Click on the  models in the right sidebar for more examples of how to apply Gemma to different vision and language tasks.

## Usage tips

- 4.3× smaller and 5.5× faster than BERT_BASE.
- MobileBERT is a thin version of BERT_LARGE, featuring bottleneck layers that reduce computational complexity while preserving model expressiveness.
- The model is deeper than BERT_BASE but with narrower layers, making it more parameter-efficient.
- Achieves competitive performance on standard NLP benchmarks
    * GLUE score: 77.7 (only 0.6 lower than BERT_BASE).
    * SQuAD v1.1/v2.0: F1 scores of 90.0/79.2 (outperforming BERT_BASE by 1.5/2.1 points).


The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

# Load MobileBERT for masked token prediction
mask_filler = pipeline("fill-mask", model="google/mobilebert-uncased")

# Example: Predict the masked word
results = mask_filler("The capital of France is [MASK].")
print(results[0]["sequence"]) # Output: "The capital of France is paris."
```
</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
model = AutoModelForMaskedLM.from_pretrained("google/mobilebert-uncased")

# Example input with a [MASK] token
text = "I want to [MASK] a coffee."
inputs = tokenizer(text, return_tensors="pt")

# Find the position of the [MASK] token
mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

# Predict the masked token
outputs = model(**inputs)
logits = outputs.logits
mask_logits = logits[0, mask_token_index, :]
predicted_token_id = torch.argmax(mask_logits, dim=-1)

# Decode the prediction
predicted_token = tokenizer.decode(predicted_token_id)
print(f"Predicted word: {predicted_token}")  # Output: "buy" or "order"
```

</hfoption>
<hfoption id="transformers-cli">

```bash
python -c "from transformers import pipeline; print(pipeline('fill-mask', model='google/mobilebert-uncased')('Artificial intelligence will [MASK] the world.')[0]['sequence'])"
```

</hfoption>
</hfoptions>


## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## MobileBertConfig

[[autodoc]] MobileBertConfig

## MobileBertTokenizer

[[autodoc]] MobileBertTokenizer

## MobileBertTokenizerFast

[[autodoc]] MobileBertTokenizerFast

## MobileBert specific outputs

[[autodoc]] models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput

[[autodoc]] models.mobilebert.modeling_tf_mobilebert.TFMobileBertForPreTrainingOutput

<frameworkcontent>
<pt>

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

</pt>
<tf>

## TFMobileBertModel

[[autodoc]] TFMobileBertModel
    - call

## TFMobileBertForPreTraining

[[autodoc]] TFMobileBertForPreTraining
    - call

## TFMobileBertForMaskedLM

[[autodoc]] TFMobileBertForMaskedLM
    - call

## TFMobileBertForNextSentencePrediction

[[autodoc]] TFMobileBertForNextSentencePrediction
    - call

## TFMobileBertForSequenceClassification

[[autodoc]] TFMobileBertForSequenceClassification
    - call

## TFMobileBertForMultipleChoice

[[autodoc]] TFMobileBertForMultipleChoice
    - call

## TFMobileBertForTokenClassification

[[autodoc]] TFMobileBertForTokenClassification
    - call

## TFMobileBertForQuestionAnswering

[[autodoc]] TFMobileBertForQuestionAnswering
    - call

</tf>
</frameworkcontent>
