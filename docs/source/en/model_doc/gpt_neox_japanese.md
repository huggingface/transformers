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
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColorF=white">

    </div>
</div>

*This model was released on 2022-07-27 and added to Hugging Face Transformers on 2022-09-14.*

# GPT-NeoX-Japanese

GPT-NeoX-Japanese, a Japanese language model based on [GPT-NeoX](./gpt_neox).
Japanese uses three types of characters (hiragana, katakana, kanji) and has a huge vocabulary. This model uses [BPEEncoder V2](https://github.com/tanreinama/Japanese-BPEEncoder_V2), a sub-word tokenizer to handle the different characters.



The model also removes some bias parameters for better performance.

You can find all the original GPT-NeoX-Japanese checkpoints under the [ABEJA](https://huggingface.co/abeja/models?search=gpt-neo-x) organization.

> [!TIP]
> This model was contributed by [Shinya Otani](https://github.com/SO0529), [Takayoshi Makabe](https://github.com/spider-man-tm), [Anuj Arora](https://github.com/Anuj040), and [Kyo Hattori](https://github.com/go5paopao) from [ABEJA, Inc.](https://www.abejainc.com/).
>
> Click on the GPT-NeoX-Japanese models in the right sidebar for more examples of how to apply GPT-NeoX-Japanese to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline
pipeline = pipeline(task="text-generation", 
                    model="abeja/gpt-neox-japanese-2.7b", dtype=torch.float16, device=0)
pipeline("人とAIが協調するためには、")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b", dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
input_ids = tokenizer("人とAIが協調するためには、", return_tensors="pt").input_ids.to(model.device)
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">
```bash
echo -e "人とAIが協調するためには、" | transformers run --task text-generation --model abeja/gpt-neox-japanese-2.7b --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)
model = AutoModelForCausalLM.from_pretrained(
    "abeja/gpt-neox-japanese-2.7b",
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
input_ids = tokenizer.encode("人とAIが協調するためには、", return_tensors="pt").to(model.device)
output = model.generate(input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("abeja/gpt-neox-japanese-2.7b")
visualizer("<img>What is shown in this image?")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/gpt_neox_japanese-attn-mask.png"/>
</div>

## Resources
Refer to the [Training a better GPT model: Learnings from PaLM](https://medium.com/ml-abeja/training-a-better-gpt-2-93b157662ae4) blog post for more details about how ABEJA trained GPT-NeoX-Japanese.

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
