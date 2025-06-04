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

# XLM

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    </div>
</div>

[XLM](https://arxiv.org/abs/1901.07291) is a transformer model pretrained using one of three objectives: causal language modeling (CLM), masked language modeling (MLM), or translation language modeling (TLM). What makes XLM unique is its ability to handle multiple languages through cross-lingual pretraining, achieving state-of-the-art results on cross-lingual classification and machine translation tasks.

You can find all the original XLM checkpoints under the [XLM](https://huggingface.co/models?search=xlm) collection.

> [!TIP]
> Click on the XLM models in the right sidebar for more examples of how to apply XLM to different cross-lingual tasks like classification, translation, and question answering.

The example below demonstrates how to use XLM for masked language modeling with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

# Initialize the pipeline with a multilingual XLM model
unmasker = pipeline("fill-mask", model="xlm-mlm-en-2048")

# Example in English
result = unmasker("Hello, I'm a [MASK] model.")
print(result)

# Example in French
result = unmasker("Bonjour, je suis un modèle [MASK].")
print(result)
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForMaskedLM.from_pretrained("xlm-mlm-en-2048")
tokenizer = AutoTokenizer.from_pretrained("xlm-mlm-en-2048")

# Prepare input
text = "Hello, I'm a [MASK] model."
inputs = tokenizer(text, return_tensors="pt")

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)

# Decode prediction
predicted_token = tokenizer.decode(predictions[0][inputs["input_ids"][0] == tokenizer.mask_token_id])
print(f"Predicted token: {predicted_token}")
```

</hfoption>
</hfoptions>

## Model Details

XLM was proposed in [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau. The model achieves state-of-the-art results on several cross-lingual tasks:

- 4.9% absolute gain in accuracy on XNLI
- 34.3 BLEU on WMT'16 German-English (unsupervised)
- 38.5 BLEU on WMT'16 Romanian-English (supervised)

### Key Features

- Supports multiple languages through cross-lingual pretraining
- Three training objectives:
  - Causal language modeling (CLM) for autoregressive generation
  - Masked language modeling (MLM) similar to BERT
  - Translation language modeling (TLM) for parallel data
- Multilingual checkpoints with language-specific parameters

## Usage Tips

- Choose the appropriate checkpoint based on your task (CLM for generation, MLM for understanding)
- For multilingual tasks, use the `lang` parameter to specify the language
- The model supports input sequences of up to 256 tokens
- For TLM, the model can leverage context from both languages in parallel data

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## XLMConfig

[[autodoc]] XLMConfig

## XLMTokenizer

[[autodoc]] XLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLM specific outputs

[[autodoc]] models.xlm.modeling_xlm.XLMForQuestionAnsweringOutput

<frameworkcontent>
<pt>

## XLMModel

[[autodoc]] XLMModel
    - forward

## XLMWithLMHeadModel

[[autodoc]] XLMWithLMHeadModel
    - forward

## XLMForSequenceClassification

[[autodoc]] XLMForSequenceClassification
    - forward

## XLMForMultipleChoice

[[autodoc]] XLMForMultipleChoice
    - forward

## XLMForTokenClassification

[[autodoc]] XLMForTokenClassification
    - forward

## XLMForQuestionAnsweringSimple

[[autodoc]] XLMForQuestionAnsweringSimple
    - forward

## XLMForQuestionAnswering

[[autodoc]] XLMForQuestionAnswering
    - forward

</pt>
<tf>

## TFXLMModel

[[autodoc]] TFXLMModel
    - call

## TFXLMWithLMHeadModel

[[autodoc]] TFXLMWithLMHeadModel
    - call

## TFXLMForSequenceClassification

[[autodoc]] TFXLMForSequenceClassification
    - call

## TFXLMForMultipleChoice

[[autodoc]] TFXLMForMultipleChoice
    - call

## TFXLMForTokenClassification

[[autodoc]] TFXLMForTokenClassification
    - call

## TFXLMForQuestionAnsweringSimple

[[autodoc]] TFXLMForQuestionAnsweringSimple
    - call

</tf>
</frameworkcontent>


