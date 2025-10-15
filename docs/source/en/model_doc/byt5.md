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
*This model was released on 2021-05-28 and added to Hugging Face Transformers on 2021-06-01 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

# ByT5

[ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://huggingface.co/papers/2105.13626) explores the use of standard Transformer architectures to process byte sequences directly, eliminating the need for tokenization. This approach offers benefits such as language-agnostic processing, robustness to noise, and reduced preprocessing complexity. The study demonstrates that byte-level models can compete with token-level models in terms of parameter count, training computational cost, and inference speed. Additionally, byte-level models show superior performance on tasks sensitive to spelling and pronunciation. The paper introduces a new set of pre-trained byte-level Transformer models based on the T5 architecture.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text2text-generation", model="google/byt5-small", dtype="auto")
pipeline("translate English to French: Plants generate energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

inputs = tokenizer("translate English to French: Plants generate energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfopton>
</hfoptions>

## Usage tips

- Use the tokenizer for batched inference and training.
- ByT5 uses top byte values (258, 257, etc.) for masking instead of sentinel tokens like `{extra_id_0}`.

## ByT5Tokenizer

[[autodoc]] ByT5Tokenizer

See [`ByT5Tokenizer`] for all details.

