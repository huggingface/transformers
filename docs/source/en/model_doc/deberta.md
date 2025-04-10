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
</div>

# DeBERTa

[DeBERTa](https://arxiv.org/abs/2006.03654) stands for *Decoding-enhanced BERT with Disentangled Attention*, and it's like BERT and RoBERTa's smarter cousin. It builds on those models but takes things up a notch with two key ideas: disentangled attention and an enhanced mask decoder. Instead of mixing everything together like BERT, DeBERTa separates a word's *content* from its *position* and processes them independently—giving it a clearer sense of what's being said and where in the sentence it's happening.

Another cool thing is how DeBERTa handles masked language modeling. Rather than relying on the traditional softmax decoder, it introduces a more powerful mechanism that makes better predictions and trains more efficiently.

Even with less training data than RoBERTa, DeBERTa manages to outperform it on several benchmarks. It's become a go-to model for high-performance language tasks, and it's especially great when you need strong accuracy without maxing out compute.

You can find all DeBERTa checkpoints here
[microsoft/deberta-base](https://huggingface.co/microsoft/deberta-base)
[microsoft/deberta-large](https://huggingface.co/microsoft/deberta-large)
[microsoft/deberta-large-mnli](https://huggingface.co/microsoft/deberta-large-mnli)


> [!TIP]
> Click on the DeBERTa models in the right sidebar for more examples of how to apply DeBERTa to different language tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline>

```py
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="microsoft/deberta-base-mnli"
)

classifier({
    "text": "A soccer game with multiple people playing.",
    "text_pair": "Some people are playing a sport."
})
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "microsoft/deberta-base-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer(
    "A soccer game with multiple people playing.",
    "Some people are playing a sport.",
    return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class = logits.argmax().item()

labels = ["contradiction", "neutral", "entailment"]
print(f"The predicted relation is: {labels[predicted_class]}")

```

</hfoption>
<hfoption id="transformers-cli">

```bash
echo -e '{"text": "A soccer game with multiple people playing.", "text_pair": "Some people are playing a sport."}' | transformers-cli run --task text-classification --model microsoft/deberta-base-mnli
```

</hfoption>
</hfoptions

## Notes
- DeBERTa uses **relative position embeddings**, so it does **not require right-padding** like BERT.
- It separates **content** and **position** information in its attention mechanism, which improves contextual understanding.
- For best results, use DeBERTa on **sentence-level or sentence-pair classification tasks** like MNLI, RTE, or SST-2.
- If you're using DeBERTa for token-level tasks like masked language modeling, make sure to load a checkpoint that was specifically pre-trained or fine-tuned for it

## DebertaConfig

[[autodoc]] DebertaConfig

## DebertaTokenizer

[[autodoc]] DebertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## DebertaTokenizerFast

[[autodoc]] DebertaTokenizerFast
    - build_inputs_with_special_tokens
    - create_token_type_ids_from_sequences

<frameworkcontent>
<pt>

## DebertaModel

[[autodoc]] DebertaModel
    - forward

## DebertaPreTrainedModel

[[autodoc]] DebertaPreTrainedModel

## DebertaForMaskedLM

[[autodoc]] DebertaForMaskedLM
    - forward

## DebertaForSequenceClassification

[[autodoc]] DebertaForSequenceClassification
    - forward

## DebertaForTokenClassification

[[autodoc]] DebertaForTokenClassification
    - forward

## DebertaForQuestionAnswering

[[autodoc]] DebertaForQuestionAnswering
    - forward

</pt>
<tf>

## TFDebertaModel

[[autodoc]] TFDebertaModel
    - call

## TFDebertaPreTrainedModel

[[autodoc]] TFDebertaPreTrainedModel
    - call

## TFDebertaForMaskedLM

[[autodoc]] TFDebertaForMaskedLM
    - call

## TFDebertaForSequenceClassification

[[autodoc]] TFDebertaForSequenceClassification
    - call

## TFDebertaForTokenClassification

[[autodoc]] TFDebertaForTokenClassification
    - call

## TFDebertaForQuestionAnswering

[[autodoc]] TFDebertaForQuestionAnswering
    - call

</tf>
</frameworkcontent>

