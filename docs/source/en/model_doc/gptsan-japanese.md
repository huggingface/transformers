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
*This model was released on 2023-02-07 and added to Hugging Face Transformers on 2023-06-20.*

> [!WARNING]
> This model is in maintenance mode only, we don't accept any new PRs changing its code. If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.

# GPTSAN-japanese

[GPTSAN-japanese](https://huggingface.co/Tanrei/GPTSAN-japanese) is a Japanese language model based on a Switch Transformer architecture and built as a Prefix-LM (hybrid between masked LM and autoregressive) so it can do both continuation (generation) and fill-in (masked) tasks. It incorporates a special “Spout vector” input (128-dim, processed via an 8-layer FFN) which can bias the generation behavior. Its default config uses 1,024 hidden dim, 8,192 intermediate FF dim, 10 switch layers, 16 experts per switch layer, and vocab size 36,000.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Tanrei/GPTSAN-japanese", dtype="auto",)
pipeline("植物は光合成と呼ばれる過程を通じてエネルギーを作り出します。")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
model = AutoModelForCausalLM.from_pretrained("Tanrei/GPTSAN-japanese", dtype="auto",)

inputs = tokenizer("植物は光合成と呼ばれる過程を通じてエネルギーを作り出します。", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- GPTSAN has unique features including a Prefix-LM model structure. It works as a shifted Masked Language Model for Prefix Input tokens. Un-prefixed inputs behave like normal generative models.
- The Spout vector is a GPTSAN-specific input. Spout is pre-trained with random inputs, but you can specify a class of text or an arbitrary vector during fine-tuning. This indicates the tendency of generated text.
- GPTSAN has a sparse Feed Forward based on Switch-Transformer. You can add other layers and train them partially. See the original GPTSAN repository for details.
- GPTSAN uses the Prefix-LM structure from the T5 paper. The original GPTSAN repository calls it hybrid. In GPTSAN, the Prefix part can be specified with any length. Arbitrary lengths can be specified differently for each batch.
- This length applies to the text entered in `prefix_text` for the tokenizer. The tokenizer returns the mask of the Prefix part as `token_type_ids`. The model treats the part where `token_type_ids` is 1 as a Prefix part, meaning the input can refer to both tokens before and after.
- Specifying the Prefix part is done with a mask passed to self-attention. When `token_type_ids=None` or all zero, it's equivalent to regular causal mask.
- A Spout Vector is a special vector for controlling text generation. This vector is treated as the first embedding in self-attention to bring extraneous attention to generated tokens.
- In the pre-trained model from Tanrei/GPTSAN-japanese, the Spout Vector is a 128-dimensional vector that passes through 8 fully connected layers and is projected into the space acting as external attention. The Spout Vector projected by the fully connected layer is split to be passed to all self-attentions.

## GPTSanJapaneseConfig

[[autodoc]] GPTSanJapaneseConfig

## GPTSanJapaneseTokenizer

[[autodoc]] GPTSanJapaneseTokenizer

## GPTSanJapaneseModel

[[autodoc]] GPTSanJapaneseModel

## GPTSanJapaneseForConditionalGeneration

[[autodoc]] GPTSanJapaneseForConditionalGeneration
    - forward
