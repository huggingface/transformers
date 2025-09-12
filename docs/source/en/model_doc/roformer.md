<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-04-20 and added to Hugging Face Transformers on 2021-05-20.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# RoFormer

[RoFormer](https://huggingface.co/papers/2104.09864) introduces Rotary Position Embedding (RoPE) to encode token positions by rotating the inputs in 2D space. This allows a model to track absolute positions and model relative relationships. RoPE can scale to longer sequences, account for the natural decay of token dependencies, and works with the more efficient linear self-attention.

You can find all the RoFormer checkpoints on the [Hub](https://huggingface.co/models?search=roformer).

> [!TIP]
> Click on the RoFormer models in the right sidebar for more examples of how to apply RoFormer to different language tasks.

The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
# uncomment to install rjieba which is needed for the tokenizer
# !pip install rjieba
import torch
from transformers import pipeline

pipe = pipeline(
    task="fill-mask",
    model="junnyu/roformer_chinese_base",
    dtype=torch.float16,
    device=0
)
output = pipe("水在零度时会[MASK]")
print(output)
```

</hfoption>
<hfoption id="AutoModel">

```py
# uncomment to install rjieba which is needed for the tokenizer
# !pip install rjieba
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained(
    "junnyu/roformer_chinese_base", dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("junnyu/roformer_chinese_base")

input_ids = tokenizer("水在零度时会[MASK]", return_tensors="pt").to(model.device)
outputs = model(**input_ids)
decoded = tokenizer.batch_decode(outputs.logits.argmax(-1), skip_special_tokens=True)
print(decoded)
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "水在零度时会[MASK]" | transformers-cli run --task fill-mask --model junnyu/roformer_chinese_base --device 0
```

</hfoption>
</hfoptions>

## Notes

- The current RoFormer implementation is an encoder-only model. The original code can be found in the [ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer) repository.

## RoFormerConfig

[[autodoc]] RoFormerConfig

## RoFormerTokenizer

[[autodoc]] RoFormerTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RoFormerTokenizerFast

[[autodoc]] RoFormerTokenizerFast
    - build_inputs_with_special_tokens

## RoFormerModel

[[autodoc]] RoFormerModel
    - forward

## RoFormerForCausalLM

[[autodoc]] RoFormerForCausalLM
    - forward

## RoFormerForMaskedLM

[[autodoc]] RoFormerForMaskedLM
    - forward

## RoFormerForSequenceClassification

[[autodoc]] RoFormerForSequenceClassification
    - forward

## RoFormerForMultipleChoice

[[autodoc]] RoFormerForMultipleChoice
    - forward

## RoFormerForTokenClassification

[[autodoc]] RoFormerForTokenClassification
    - forward

## RoFormerForQuestionAnswering

[[autodoc]] RoFormerForQuestionAnswering
    - forward
