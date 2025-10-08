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
*This model was released on 2022-07-27 and added to Hugging Face Transformers on 2022-09-14 and contributed by [SO0529](https://github.com/SO0529), [spider-man-tm](https://github.com/spider-man-tm), [Anuj040](https://github.com/Anuj040), and [go5paopao](https://github.com/go5paopao).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# GPT-NeoX-Japanese

GPT-NeoX-Japanese is a GPT-NeoX model trained on Japanese.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="abeja/gpt-neox-japanese-2.7b", dtype="auto",)
pipeline("植物は光合成と呼ばれる過程を通じてエネルギーを作り出します。")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
model = AutoModelForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b", dtype="auto",)

inputs = tokenizer("植物は光合成と呼ばれる過程を通じてエネルギーを作り出します。", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## GPTNeoXJapaneseConfig

[[autodoc]] GPTNeoXJapaneseConfig

## GPTNeoXJapaneseTokenizer

[[autodoc]] GPTNeoXJapaneseTokenizer

## GPTNeoXJapaneseModel

[[autodoc]] GPTNeoXJapaneseModel
    - forward

## GPTNeoXJapaneseForCausalLM

[[autodoc]] GPTNeoXJapaneseForCausalLM
    - forward
