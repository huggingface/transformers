<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-01-11 and added to Hugging Face Transformers on 2022-11-15 and contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).*

# SwitchTransformers

[SwitchTransformers](https://huggingface.co/papers/2101.03961) employs a sparse T5 encoder-decoder architecture by replacing MLPs with a Mixture of Experts (MoE). This architecture uses a routing mechanism to associate each token with one expert, a dense MLP, enabling better scaling and fine-tuning performance. During inference, only a fraction of the weights is utilized, enhancing model capacity without increasing operations. The paper simplifies the MoE routing algorithm, reduces communication and computational costs, and addresses training instability. SwitchTransformers achieve up to a 7x increase in pre-training speed with the same resources and demonstrate gains in multilingual settings, outperforming mT5-Base across 101 languages. The model also supports pre-training of trillion-parameter models on the Colossal Clean Crawled Corpus, offering a 4x speedup over T5-XXL.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text2text-generation", model="google/switch-base-8", dtype="auto",)
pipeline("Plants create energy through a process known as <extra_id_0>.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as <extra_id_0>.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## SwitchTransformersConfig

[[autodoc]] SwitchTransformersConfig

## SwitchTransformersTop1Router

[[autodoc]] SwitchTransformersTop1Router
    - forward

## SwitchTransformersSparseMLP

[[autodoc]] SwitchTransformersSparseMLP
    - forward

## SwitchTransformersModel

[[autodoc]] SwitchTransformersModel
    - forward

## SwitchTransformersForConditionalGeneration

[[autodoc]] SwitchTransformersForConditionalGeneration
    - forward

## SwitchTransformersEncoderModel

[[autodoc]] SwitchTransformersEncoderModel
    - forward

