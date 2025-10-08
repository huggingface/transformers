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
*This model was released on 2020-01-13 and added to Hugging Face Transformers on 2020-11-16 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

# Reformer

[Reformer: The Efficient Transformer](https://huggingface.co/papers/2001.04451) introduces techniques to enhance Transformer efficiency. It replaces dot-product attention with locality-sensitive hashing, reducing complexity from O(L^2) to O(Llog(L)). Additionally, it employs reversible residual layers, enabling memory savings by storing activations once during training. This results in a model that matches Transformer performance while being more memory-efficient and faster on long sequences.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/reformer-crime-and-punishment", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("google/reformer-crime-and-punishment", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## ReformerConfig

[[autodoc]] ReformerConfig

## ReformerTokenizer

[[autodoc]] ReformerTokenizer
    - save_vocabulary

## ReformerTokenizerFast

[[autodoc]] ReformerTokenizerFast

## ReformerModel

[[autodoc]] ReformerModel
    - forward

## ReformerModelWithLMHead

[[autodoc]] ReformerModelWithLMHead
    - forward

## ReformerForMaskedLM

[[autodoc]] ReformerForMaskedLM
    - forward

## ReformerForSequenceClassification

[[autodoc]] ReformerForSequenceClassification
    - forward

## ReformerForQuestionAnswering

[[autodoc]] ReformerForQuestionAnswering
    - forward

