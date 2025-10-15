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
*This model was released on 2024-03-13 and added to Hugging Face Transformers on 2024-02-21 and contributed by [ArthurZ](https://huggingface.co/ArthurZ), [ybelkada](https://huggingface.co/ybelkada), [sanchit-gandhi](https://huggingface.co/sanchit-gandhi), and [pcuenq](https://huggingface.co/pcuenq).*

# Gemma

[Gemma](https://huggingface.co/papers/2403.08295) is a family of lightweight open-source language models derived from Gemini research, available in 2-billion and 7-billion parameter versions with both pretrained and fine-tuned checkpoints. The models achieve strong performance on academic benchmarks for language understanding, reasoning, and safety, outperforming comparable open models on 11 of 18 text-based tasks. The work includes comprehensive evaluations of model safety and responsible deployment, alongside detailed documentation of model development. Gemma emphasizes the importance of responsible LLM release to support safer frontier models and future innovations.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- Gemma models support standard kv-caching used in transformer-based language models. Use the default [`DynamicCache`] instance or a tuple of tensors for past key values during generation. This works with typical autoregressive generation workflows.

## GemmaConfig

[[autodoc]] GemmaConfig

## GemmaTokenizer

[[autodoc]] GemmaTokenizer

## GemmaTokenizerFast

[[autodoc]] GemmaTokenizerFast

## GemmaModel

[[autodoc]] GemmaModel
    - forward

## GemmaForCausalLM

[[autodoc]] GemmaForCausalLM
    - forward

## GemmaForSequenceClassification

[[autodoc]] GemmaForSequenceClassification
    - forward

## GemmaForTokenClassification

[[autodoc]] GemmaForTokenClassification
    - forward

