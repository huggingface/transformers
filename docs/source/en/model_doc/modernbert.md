<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# ModernBERT

[ModernBERT](https://huggingface.co/papers/2412.13663) is an improved succesor model built on the traditional encoder architecture. It builds on BERT and implements many modern architectural improvements which have been developed since its original release, such as:
- [Rotary Positional Embeddings](https://huggingface.co/blog/designing-positional-encoding) to support sequences of up to 8192 tokens.
- [Unpadding](https://arxiv.org/abs/2208.08124) to ensure no compute is wasted on padding tokens, speeding up processing time for batches with mixed-length sequences.
- [GeGLU](https://arxiv.org/abs/2002.05202) Replacing the original MLP layers with GeGLU layers, shown to improve performance.
- [Alternating Attention](https://arxiv.org/abs/2004.05150v2) where most attention layers employ a sliding window of 128 tokens, with Global Attention only used every 3 layers.
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) to speed up processing.
- A model designed following recent [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489), ensuring maximum efficiency across inference GPUs.
- Modern training data scales (2 trillion tokens) and mixtures (including code ande math data)

The original code can be found [here](https://github.com/answerdotai/modernbert).

The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="answerdotai/ModernBERT-base",
    torch_dtype=torch.float16,
    device=0
)
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "answerdotai/ModernBERT-base",
)
model = AutoModelForMaskedLM.from_pretrained(
    "answerdotai/ModernBERT-base",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

</hfoption>
<hfoption id="transformers-cli">

```bash
echo -e "Plants create [MASK] through a process known as photosynthesis." | transformers-cli run --task fill-mask --model answerdotai/ModernBERT-base --device 0
```

</hfoption>
</hfoptions>

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ModernBert.

<PipelineTag pipeline="text-classification"/>

- A notebook on how to [finetune for General Language Understanding Evaluation (GLUE) with Transformers](https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/finetune_modernbert_on_glue.ipynb), also available as a Google Colab [notebook](https://colab.research.google.com/github/AnswerDotAI/ModernBERT/blob/main/examples/finetune_modernbert_on_glue.ipynb). 🌎

<PipelineTag pipeline="sentence-similarity"/>

- A script on how to [finetune for text similarity or information retrieval with Sentence Transformers](https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/train_st.py). 🌎
- A script on how to [finetune for information retrieval with PyLate](https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/train_pylate.py). 🌎

<PipelineTag pipeline="fill-mask"/>

- [Masked language modeling task guide](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- [`ModernBertForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) and [colab notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).

## ModernBertConfig

[[autodoc]] ModernBertConfig

<frameworkcontent>
<pt>

## ModernBertModel

[[autodoc]] ModernBertModel
    - forward

## ModernBertForMaskedLM

[[autodoc]] ModernBertForMaskedLM
    - forward

## ModernBertForSequenceClassification

[[autodoc]] ModernBertForSequenceClassification
    - forward

## ModernBertForTokenClassification

[[autodoc]] ModernBertForTokenClassification
    - forward

## ModernBertForQuestionAnswering

[[autodoc]] ModernBertForQuestionAnswering
    - forward

### Usage tips

The ModernBert model can be fine-tuned using the HuggingFace Transformers library with its [official script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py) for question-answering tasks.


</pt>
</frameworkcontent>
