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
*This model was released on 2020-06-05 and added to Hugging Face Transformers on 2020-11-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# DeBERTa

[DeBERTa](https://huggingface.co/papers/2006.03654) improves the pretraining efficiency of BERT and RoBERTa with two key ideas, disentangled attention and an enhanced mask decoder. Instead of mixing everything together like BERT, DeBERTa separates a word's *content* from its *position* and processes them independently. This gives it a clearer sense of what's being said and where in the sentence it's happening.

The enhanced mask decoder replaces the traditional softmax decoder to make better predictions.

Even with less training data than RoBERTa, DeBERTa manages to outperform it on several benchmarks.

You can find all the original DeBERTa checkpoints under the [Microsoft](https://huggingface.co/microsoft?search_models=deberta) organization.


> [!TIP]
> Click on the DeBERTa models in the right sidebar for more examples of how to apply DeBERTa to different language tasks.

The example below demonstrates how to classify text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="microsoft/deberta-base-mnli",
    device=0,
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "microsoft/deberta-base-mnli"
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base-mnli", device_map="auto")

inputs = tokenizer(
    "A soccer game with multiple people playing.",
    "Some people are playing a sport.",
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class = logits.argmax().item()

labels = ["contradiction", "neutral", "entailment"]
print(f"The predicted relation is: {labels[predicted_class]}")

```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e '{"text": "A soccer game with multiple people playing.", "text_pair": "Some people are playing a sport."}' | transformers run --task text-classification --model microsoft/deberta-base-mnli --device 0
```

</hfoption>
</hfoptions>

## Notes
- DeBERTa uses **relative position embeddings**, so it does not require **right-padding** like BERT.
- For best results, use DeBERTa on sentence-level or sentence-pair classification tasks like MNLI, RTE, or SST-2.
- If you're using DeBERTa for token-level tasks like masked language modeling, make sure to load a checkpoint specifically pretrained or fine-tuned for token-level tasks.

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
