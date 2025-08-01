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
        <img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,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">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# XLM-RoBERTa

[XLM-RoBERTa](https://huggingface.co/papers/1911.02116) is a large multilingual masked language model trained on 2.5TB of filtered CommonCrawl data across 100 languages. It shows that scaling the model provides strong performance gains on high-resource and low-resource languages. The model uses the [RoBERTa](./roberta) pretraining objectives on the [XLM](./xlm) model.

You can find all the original XLM-RoBERTa checkpoints under the [Facebook AI community](https://huggingface.co/FacebookAI) organization.

> [!TIP]
> Click on the XLM-RoBERTa models in the right sidebar for more examples of how to apply XLM-RoBERTa to different cross-lingual tasks like classification, translation, and question answering.

The example below demonstrates how to predict the `<mask>` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="FacebookAI/xlm-roberta-base",
    dtype=torch.float16,
    device=0
)
# Example in French
pipeline("Bonjour, je suis un modèle <mask>.")

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/xlm-roberta-base"
)
model = AutoModelForMaskedLM.from_pretrained(
    "FacebookAI/xlm-roberta-base",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

# Prepare input
inputs = tokenizer("Bonjour, je suis un modèle <mask>.", return_tensors="pt").to("cuda")

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
echo -e "Plants create <mask> through a process known as photosynthesis." | transformers-cli run --task fill-mask --model FacebookAI/xlm-roberta-base --device 0
```
</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [quantization guide](../quantization) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) the quantive the weights to 4 bits

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    bnb_4bit_quant_type="nf4",  # or "fp4" for float 4-bit quantization
    bnb_4bit_use_double_quant=True,  # use double quantization for better performance
)
tokenizer = AutoTokenizer.from_pretrained("facebook/xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained(
    "facebook/xlm-roberta-large",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config
)

inputs = tokenizer("Bonjour, je suis un modèle <mask>.", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Notes

- Unlike some XLM models, XLM-RoBERTa doesn't require `lang` tensors to understand what language is being used. It automatically determines the language from the input IDs

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with XLM-RoBERTa. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-classification"/>

- A blog post on how to [finetune XLM RoBERTa for multiclass classification with Habana Gaudi on AWS](https://www.philschmid.de/habana-distributed-training)
- [`XLMRobertaForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb).
- [`TFXLMRobertaForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).
- [`FlaxXLMRobertaForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb).
- [Text classification](https://huggingface.co/docs/transformers/tasks/sequence_classification) chapter of the 🤗 Hugging Face Task Guides.
- [Text classification task guide](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- [`XLMRobertaForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb).
- [`TFXLMRobertaForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).
- [`FlaxXLMRobertaForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification).
- [Token classification](https://huggingface.co/course/chapter7/2?fw=pt) chapter of the 🤗 Hugging Face Course.
- [Token classification task guide](../tasks/token_classification)

<PipelineTag pipeline="text-generation"/>

- [`XLMRobertaForCausalLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) chapter of the 🤗 Hugging Face Task Guides.
- [Causal language modeling task guide](../tasks/language_modeling)

<PipelineTag pipeline="fill-mask"/>

- [`XLMRobertaForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFXLMRobertaForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- [`FlaxXLMRobertaForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb).
- [Masked language modeling](https://huggingface.co/course/chapter7/3?fw=pt) chapter of the 🤗 Hugging Face Course.
- [Masked language modeling](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- [`XLMRobertaForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).
- [`TFXLMRobertaForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).
- [`FlaxXLMRobertaForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering).
- [Question answering](https://huggingface.co/course/chapter7/7?fw=pt) chapter of the 🤗 Hugging Face Course.
- [Question answering task guide](../tasks/question_answering)

**Multiple choice**

- [`XLMRobertaForMultipleChoice`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb).
- [`TFXLMRobertaForMultipleChoice`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).
- [Multiple choice task guide](../tasks/multiple_choice)

🚀 Deploy

- A blog post on how to [Deploy Serverless XLM RoBERTa on AWS Lambda](https://www.philschmid.de/multilingual-serverless-xlm-roberta-with-huggingface).

<Tip> 

This implementation is the same as RoBERTa. Refer to the [documentation of RoBERTa](roberta) for usage examples as well as the information relative to the inputs and outputs.
</Tip>

## XLMRobertaConfig

[[autodoc]] XLMRobertaConfig

## XLMRobertaTokenizer

[[autodoc]] XLMRobertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLMRobertaTokenizerFast

[[autodoc]] XLMRobertaTokenizerFast

<frameworkcontent>
<pt>

## XLMRobertaModel

[[autodoc]] XLMRobertaModel
    - forward

## XLMRobertaForCausalLM

[[autodoc]] XLMRobertaForCausalLM
    - forward

## XLMRobertaForMaskedLM

[[autodoc]] XLMRobertaForMaskedLM
    - forward

## XLMRobertaForSequenceClassification

[[autodoc]] XLMRobertaForSequenceClassification
    - forward

## XLMRobertaForMultipleChoice

[[autodoc]] XLMRobertaForMultipleChoice
    - forward

## XLMRobertaForTokenClassification

[[autodoc]] XLMRobertaForTokenClassification
    - forward

## XLMRobertaForQuestionAnswering

[[autodoc]] XLMRobertaForQuestionAnswering
    - forward

</pt>
<tf>

## TFXLMRobertaModel

[[autodoc]] TFXLMRobertaModel
    - call

## TFXLMRobertaForCausalLM

[[autodoc]] TFXLMRobertaForCausalLM
    - call

## TFXLMRobertaForMaskedLM

[[autodoc]] TFXLMRobertaForMaskedLM
    - call

## TFXLMRobertaForSequenceClassification

[[autodoc]] TFXLMRobertaForSequenceClassification
    - call

## TFXLMRobertaForMultipleChoice

[[autodoc]] TFXLMRobertaForMultipleChoice
    - call

## TFXLMRobertaForTokenClassification

[[autodoc]] TFXLMRobertaForTokenClassification
    - call

## TFXLMRobertaForQuestionAnswering

[[autodoc]] TFXLMRobertaForQuestionAnswering
    - call

</tf>
<jax>

## FlaxXLMRobertaModel

[[autodoc]] FlaxXLMRobertaModel
    - __call__

## FlaxXLMRobertaForCausalLM

[[autodoc]] FlaxXLMRobertaForCausalLM
    - __call__

## FlaxXLMRobertaForMaskedLM

[[autodoc]] FlaxXLMRobertaForMaskedLM
    - __call__

## FlaxXLMRobertaForSequenceClassification

[[autodoc]] FlaxXLMRobertaForSequenceClassification
    - __call__

## FlaxXLMRobertaForMultipleChoice

[[autodoc]] FlaxXLMRobertaForMultipleChoice
    - __call__

## FlaxXLMRobertaForTokenClassification

[[autodoc]] FlaxXLMRobertaForTokenClassification
    - __call__

## FlaxXLMRobertaForQuestionAnswering

[[autodoc]] FlaxXLMRobertaForQuestionAnswering
    - __call__

</jax>
</frameworkcontent>