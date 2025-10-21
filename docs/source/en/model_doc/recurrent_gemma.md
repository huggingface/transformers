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
*This model was released on 2024-04-11 and added to Hugging Face Transformers on 2024-04-10 and contributed by [ArthurZ](https://huggingface.co/ArthurZ).*

# RecurrentGemma

[RecurrentGemma](https://huggingface.co/papers/RecurrentGemma:MovingPastTransformersForEfficientOpenLanguageModels) utilizes Google’s Griffin architecture, integrating linear recurrences with local attention to enhance language processing. It features a fixed-sized state, optimizing memory usage and facilitating efficient inference on extended sequences. Available in both pre-trained and instruction-tuned variants with 2B non-embedding parameters, RecurrentGemma matches Gemma-2B’s performance despite being trained on a smaller dataset.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/recurrentgemma-2b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/recurrentgemma-2b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## RecurrentGemmaConfig

[[autodoc]] RecurrentGemmaConfig

## RecurrentGemmaModel

[[autodoc]] RecurrentGemmaModel
    - forward

## RecurrentGemmaForCausalLM

[[autodoc]] RecurrentGemmaForCausalLM
    - forward

