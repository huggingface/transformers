<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on 2021-04-20 and added to Hugging Face Transformers on 2026-01-24.*


# NomicBERT

## Overview

The NomicBERT model currently has no academic papers specifically written about it, however, the [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) card clearly describes the model’s architecture and training approach: it extends BERT to a 2048 token context length, and modifies the BERT training procedure. Notable changes include: 

- Use [Rotary Position Embeddings](https://huggingface.co/papers/2104.09864.pdf) to allow for context length extrapolation.
- Use SwiGLU activations, which have [been shown](https://huggingface.co/papers/2002.05202) to [improve model performance](https://www.databricks.com/blog/mosaicbert)
- No dropout

> [!TIP]
> - NomicBERT can handle very long sequences efficiently (up to 2048 tokens by default).
> - For masked language modeling, use `NomicBertForMaskedLM`.
> - Use smaller configs for testing locally to save memory and speed up unit tests.
> - Supports various heads: classification, QA, token classification, multiple choice, etc.


This model was contributed by community members ([Sonny Cooper](https://github.com/ed22699)).
The original code for nomic-embed-text-v1.5 can be found [here](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).

## Usage examples
The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="nomic-ai/nomic-embed-text-v1.5",
    dtype=torch.float16,
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
    "google-bert/bert-base-uncased",
)
model = AutoModelForMaskedLM.from_pretrained(
    "nomic-ai/nomic-embed-text-v1.5",
    dtype=torch.float16,
    device_map="auto"
)
inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt").to(model.device)

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
echo -e "Plants create [MASK] through a process known as photosynthesis." | transformers run --task fill-mask --model nomic-ai/nomic-bert-2048 --device 0
```

</hfoption>
</hfoptions>

## Notes

- For extremely long sequences, consider batching or gradient checkpointing to save memory.

## NomicBertConfig

[[autodoc]] NomicBertConfig

## NomicBertForMaskedLM

[[autodoc]] NomicBertForMaskedLM

## NomicBertForMultipleChoice

[[autodoc]] NomicBertForMultipleChoice

## NomicBertForNextSentencePrediction

[[autodoc]] NomicBertForNextSentencePrediction

## NomicBertForPreTraining

[[autodoc]] NomicBertForPreTraining

## NomicBertForQuestionAnswering

[[autodoc]] NomicBertForQuestionAnswering

## NomicBertForSequenceClassification

[[autodoc]] NomicBertForSequenceClassification

## NomicBertForTokenClassification

[[autodoc]] NomicBertForTokenClassification

## NomicBertModel

[[autodoc]] NomicBertModel
    - forward
