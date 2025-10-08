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
*This model was released on 2023-05-05 and added to Hugging Face Transformers on 2023-07-25.*


# MPT

[MPT](https://www.databricks.com/blog/mpt-7b) is a 6.7-billion-parameter decoder-style transformer developed by MosaicML, trained from scratch on 1 trillion tokens of text and code over 9.5 days with zero human intervention. It is fully open-source and commercially usable, featuring FlashAttention for fast training and inference, and ALiBi to handle extremely long context lengths up to 84k tokens. MosaicML also released finetuned variants—Instruct, Chat, and StoryWriter-65k+—to demonstrate specialized capabilities. The model was rigorously benchmarked and matches the quality of LLaMA-7B while offering easier deployment, licensing for commercial use, and highly efficient training code.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="mosaicml/mpt-7b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b")
model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## MptConfig

[[autodoc]] MptConfig
    - all

## MptModel

[[autodoc]] MptModel
    - forward

## MptForCausalLM

[[autodoc]] MptForCausalLM
    - forward

## MptForSequenceClassification

[[autodoc]] MptForSequenceClassification
    - forward

## MptForTokenClassification

[[autodoc]] MptForTokenClassification
    - forward

## MptForQuestionAnswering

[[autodoc]] MptForQuestionAnswering
    - forward

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="mosaicml/mpt-7b", dtype="auto")
pipeline("The future of artificial intelligence is")
```
