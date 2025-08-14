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

[MobileBERT](https://huggingface.co/papers/2004.02984) is a lightweight and efficient variant of BERT, specifically designed for resource-limited devices such as mobile phones. It retains BERT's architecture but significantly reduces model size and inference latency while maintaining strong performance on NLP tasks. MobileBERT achieves this through a bottleneck structure and carefully balanced self-attention and feedforward networks. The model is trained by knowledge transfer from a large BERT model with an inverted bottleneck structure.

You can find the original MobileBERT checkpoint under the [Google](https://huggingface.co/google/mobilebert-uncased) organization.
> [!TIP]
> Click on the MobileBERT models in the right sidebar for more examples of how to apply MobileBERT to different language tasks.

The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="google/mobilebert-uncased",
    dtype=torch.float16,
    device=0
)
pipeline("The capital of France is [MASK].")
```
</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/mobilebert-uncased",
)
model = AutoModelForMaskedLM.from_pretrained(
    "google/mobilebert-uncased",
    dtype=torch.float16,
    device_map="auto",
)
inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "The capital of France is [MASK]." | transformers run --task fill-mask --model google/mobilebert-uncased --device 0
```

</hfoption>
</hfoptions>


## Notes

- Inputs should be padded on the right because BERT uses absolute position embeddings.

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
