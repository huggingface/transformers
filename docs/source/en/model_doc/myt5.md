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
*This model was released on 2024-03-15 and added to Hugging Face Transformers on 2024-10-06 and contributed by [Tomlim](https://huggingface.co/Tomlim).*

# myt5

[myt5](https://huggingface.co/papers/2403.10691) is a multilingual language model based on the T5 architecture. It employs a morphologically-driven byte (MYTE) representation, using codepoints corresponding to morphemes instead of characters. This approach addresses disparities in text encoding across diverse languages by producing shorter, more consistent encodings, particularly benefiting non-European languages and non-Latin scripts. MYTE leverages unsupervised morphological segmentation to create morpheme inventories for 99 languages, improving multilingual language model performance and reducing perplexity gaps.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text2text-generation", model="Tomlim/myt5-base", dtype="auto")
pipeline("""
Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.
"""
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("Tomlim/myt5-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Tomlim/myt5-base")

text="""
Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.
"""
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfopton>
</hfoptions>

## MyT5Tokenizer

[[autodoc]] MyT5Tokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary



