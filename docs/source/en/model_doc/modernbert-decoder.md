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
*This model was released on 2024-12-18 and added to Hugging Face Transformers on 2025-07-15 and contributed by [orionweller](https://huggingface.co/orionweller).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# ModernBERT Decoder

ModernBERT Decoder shares the same architecture as ModernBERT but it's trained from scratch according to [Seq vs Seq: An Open Suite of Paired Encoders and Decoders](https://huggingface.co/papers/2507.11412). The paper introduces the Ettin suite, a set of paired encoder-only and decoder-only language models ranging from 17 million to 1 billion parameters, trained on up to 2 trillion tokens using identical training recipes. These models achieve state-of-the-art performance in their respective categories, surpassing ModernBERT for encoder tasks and Llama 3.2 and SmolLM2 for decoder tasks. Encoder-only models perform best on classification and retrieval tasks, while decoders excel at generative tasks, and attempts to adapt one architecture to the other's task via continued training are less effective. All training data, checkpoints, and artifacts are open-sourced to facilitate reproducibility and further research.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="jhu-clsp/ettin-decoder-17m", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-17m")
model = AutoModelForCausalLM.from_pretrained("jhu-clsp/ettin-decoder-17m", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## ModernBertDecoderConfig

[[autodoc]] ModernBertDecoderConfig

## ModernBertDecoderModel

[[autodoc]] ModernBertDecoderModel
    - forward

## ModernBertDecoderForCausalLM

[[autodoc]] ModernBertDecoderForCausalLM
    - forward

## ModernBertDecoderForSequenceClassification

[[autodoc]] ModernBertDecoderForSequenceClassification
    - forward