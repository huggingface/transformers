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
*This model was released on 2023-04-18 and added to Hugging Face Transformers on 2023-07-03 and contributed by [agemagician](https://huggingface.co/agemagician) and [stefan-it](https://huggingface.co/stefan-it).*

# UMT5

[UMT5](https://huggingface.co/papers/2304.09151) introduces UniMax, a new sampling method for multilingual large language model pretraining that provides more balanced coverage of high-resource languages while preventing overfitting on low-resource ones by capping repeat usage per language. The authors compare UniMax against standard temperature-based sampling through extensive ablation studies across multiple model sizes and multilingual benchmarks. Results show that UniMax consistently outperforms temperature-based approaches, with benefits scaling as model size increases. Additionally, they release a refreshed mC4 corpus containing 29 trillion characters across 107 languages and pretrained umT5 checkpoints trained with UniMax.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text2text-generation", model="google/umt5-small", dtype="auto")
pipeline("""
Plants are <extra_id_0> organisms that produce their own food using a method called photosynthesis.
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

model = AutoModelForSeq2SeqLM.from_pretrained("google/umt5-small", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")

text="""
Plants are <extra_id_0> organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.
"""
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfopton>
</hfoptions>

## UMT5Config

[[autodoc]] UMT5Config

## UMT5Model

[[autodoc]] UMT5Model
    - forward

## UMT5ForConditionalGeneration

[[autodoc]] UMT5ForConditionalGeneration
    - forward

## UMT5EncoderModel

[[autodoc]] UMT5EncoderModel
    - forward

## UMT5ForSequenceClassification

[[autodoc]] UMT5ForSequenceClassification
    - forward

## UMT5ForTokenClassification

[[autodoc]] UMT5ForTokenClassification
    - forward

## UMT5ForQuestionAnswering

[[autodoc]] UMT5ForQuestionAnswering
    - forward

