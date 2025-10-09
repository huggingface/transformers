<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-07-11 and added to Hugging Face Transformers on 2023-03-27 and contributed by [ArthurZ](https://huggingface.co/ArthurZ).*

# NLLB-MOE

[NLLB](https://huggingface.co/papers/2207.04672) addresses the challenge of translating low-resource languages by developing a conditional compute model based on Sparsely Gated Mixture of Experts. This model is trained using novel data mining techniques and includes architectural and training improvements to handle thousands of translation tasks. Evaluation on the Flores-200 benchmark shows a 44% BLEU improvement over previous state-of-the-art models, while also assessing translation safety with a toxicity benchmark.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="translation", model="facebook/nllb-moe-54b", src_lang="eng_Latn", tgt_lang="fra_Latn", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-moe-54b", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"))
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

</hfoption>
</hfoptions>


## NllbMoeConfig

[[autodoc]] NllbMoeConfig

## NllbMoeTop2Router

[[autodoc]] NllbMoeTop2Router
    - route_tokens
    - forward

## NllbMoeSparseMLP

[[autodoc]] NllbMoeSparseMLP
    - forward

## NllbMoeModel

[[autodoc]] NllbMoeModel
    - forward

## NllbMoeForConditionalGeneration

[[autodoc]] NllbMoeForConditionalGeneration
    - forward

