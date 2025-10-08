<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-10-02 and added to Hugging Face Transformers on 2021-05-03 and contributed by [ikuyamada](https://huggingface.co/ikuyamada) and [nielsr](https://huggingface.co/nielsr).*

# LUKE

[LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://huggingface.co/papers/2010.01057) proposes a model based on RoBERTa that incorporates entity embeddings and an entity-aware self-attention mechanism. This model treats words and entities as independent tokens and outputs contextualized representations for both. It is trained using a masked language model task that includes predicting masked words and entities from a large entity-annotated Wikipedia corpus. The entity-aware self-attention mechanism considers token types when computing attention scores. The model demonstrates superior performance on various entity-related tasks, achieving state-of-the-art results on Open Entity, TACRED, CoNLL-2003, ReCoRD, and SQuAD 1.1 datasets.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="studio-ousia/luke-base", dtype="auto")
pipeline("Plants create <mask> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("studio-ousia/luke-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")

inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## LukeConfig

[[autodoc]] LukeConfig

## LukeTokenizer

[[autodoc]] LukeTokenizer
    - __call__
    - save_vocabulary

## LukeModel

[[autodoc]] LukeModel
    - forward

## LukeForMaskedLM

[[autodoc]] LukeForMaskedLM
    - forward

## LukeForEntityClassification

[[autodoc]] LukeForEntityClassification
    - forward

## LukeForEntityPairClassification

[[autodoc]] LukeForEntityPairClassification
    - forward

## LukeForEntitySpanClassification

[[autodoc]] LukeForEntitySpanClassification
    - forward

## LukeForSequenceClassification

[[autodoc]] LukeForSequenceClassification
    - forward

## LukeForMultipleChoice

[[autodoc]] LukeForMultipleChoice
    - forward

## LukeForTokenClassification

[[autodoc]] LukeForTokenClassification
    - forward

## LukeForQuestionAnswering

[[autodoc]] LukeForQuestionAnswering
    - forward

