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

# XLM-RoBERTa-XL

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

[XLM-RoBERTa-XL](https://arxiv.org/abs/2105.00572) is a large-scale multilingual language model with 3.5B parameters, trained on 100 languages. What makes XLM-RoBERTa-XL unique is its ability to achieve strong performance on both high-resource languages while greatly improving low-resource languages, outperforming both XLM-R and RoBERTa-Large on various benchmarks.

You can find all the original XLM-RoBERTa-XL checkpoints under the [XLM-RoBERTa-XL](https://huggingface.co/models?search=xlm-roberta-xl) collection.

> [!TIP]
> Click on the XLM-RoBERTa-XL models in the right sidebar for more examples of how to apply XLM-RoBERTa-XL to different cross-lingual tasks like classification, translation, and question answering.

The example below demonstrates how to use XLM-RoBERTa-XL for masked language modeling with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

# Initialize the pipeline with XLM-RoBERTa-XL
unmasker = pipeline("fill-mask", model="facebook/xlm-roberta-xl")

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
model = AutoModelForMaskedLM.from_pretrained("facebook/xlm-roberta-xl")
tokenizer = AutoTokenizer.from_pretrained("facebook/xlm-roberta-xl")

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

XLM-RoBERTa-XL was proposed in [Larger-Scale Transformers for Multilingual Masked Language Modeling](https://arxiv.org/abs/2105.00572) by Naman Goyal, Jingfei Du, Myle Ott, Giri Anantharaman, and Alexis Conneau. The model achieves state-of-the-art results on several benchmarks:

- 1.8% average accuracy improvement over XLM-R on XNLI
- Outperforms RoBERTa-Large on English GLUE tasks by 0.3% on average
- Handles 99 more languages than RoBERTa-Large while maintaining strong performance

### Key Features

- 3.5B parameter model (also available in 10.7B parameter version as XLM-R XXL)
- Trained on 100 different languages
- No language token required - automatically detects language from input
- Based on the XLM-RoBERTa architecture with increased model capacity
- Particularly effective for low-resource languages

## Usage Tips

- Unlike some XLM models, XLM-RoBERTa-XL doesn't require `lang` tensors - it automatically determines the language from input
- Due to its large size, consider using model parallelism or gradient checkpointing for training
- For inference, you may want to use quantization or model sharding to reduce memory requirements
- Supports all standard RoBERTa tasks (classification, token classification, QA, etc.)

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

<Tip>
This implementation is based on XLM-RoBERTa. Refer to the [documentation of XLM-RoBERTa](xlm-roberta) for usage examples as well as the information relative to the inputs and outputs.
</Tip>

## XLMRobertaXLConfig

[[autodoc]] XLMRobertaXLConfig

## XLMRobertaXLModel

[[autodoc]] XLMRobertaXLModel
    - forward

## XLMRobertaXLForCausalLM

[[autodoc]] XLMRobertaXLForCausalLM
    - forward

## XLMRobertaXLForMaskedLM

[[autodoc]] XLMRobertaXLForMaskedLM
    - forward

## XLMRobertaXLForSequenceClassification

[[autodoc]] XLMRobertaXLForSequenceClassification
    - forward

## XLMRobertaXLForMultipleChoice

[[autodoc]] XLMRobertaXLForMultipleChoice
    - forward

## XLMRobertaXLForTokenClassification

[[autodoc]] XLMRobertaXLForTokenClassification
    - forward

## XLMRobertaXLForQuestionAnswering

[[autodoc]] XLMRobertaXLForQuestionAnswering
    - forward
