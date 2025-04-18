<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    </div>
</div>

# Longformer

[Longformer](https://huggingface.co/papers/2004.05150) is a transformer model designed for long-document processing. It introduces a novel attention mechanism that scales linearly with sequence length by combining local windowed attention with task-specific global attention, enabling efficient processing of documents with thousands of tokens.

> [!TIP]
> Click on the Longformer models in the right sidebar for more examples of how to apply Longformer to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`] and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="allenai/longformer-base-4096",
    torch_dtype=torch.float16,
    device=0
)
pipeline("""San Francisco 49ers cornerback Shawntae Spencer will miss the rest of the <mask> with a torn ligament in his left knee.
Spencer, a fifth-year pro, will be placed on injured reserve soon after undergoing surgery Wednesday to repair the ligament. He injured his knee late in the 49ers’ road victory at Seattle on Sept. 14, and missed last week’s victory over Detroit.
Tarell Brown and Donald Strickland will compete to replace Spencer with the 49ers, who kept 12 defensive backs on their 53-man roster to start the season. Brown, a second-year pro, got his first career interception last weekend while filling in for Strickland, who also sat out with a knee injury.""")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = AutoModelForMaskedLM.from_pretrained("allenai/longformer-base-4096")

text = (
"""
San Francisco 49ers cornerback Shawntae Spencer will miss the rest of the <mask> with a torn ligament in his left knee.
Spencer, a fifth-year pro, will be placed on injured reserve soon after undergoing surgery Wednesday to repair the ligament. He injured his knee late in the 49ers’ road victory at Seattle on Sept. 14, and missed last week’s victory over Detroit.
Tarell Brown and Donald Strickland will compete to replace Spencer with the 49ers, who kept 12 defensive backs on their 53-man roster to start the season. Brown, a second-year pro, got his first career interception last weekend while filling in for Strickland, who also sat out with a knee injury.
"""
)

input_ids = tokenizer([text], return_tensors="pt")["input_ids"]
logits = model(input_ids).logits

masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
probs = logits[0, masked_index].softmax(dim=0)
values, predictions = probs.topk(5)
tokenizer.decode(predictions).split()
```

</hfoption>
<hfoption id="transformers-cli">

```bash
echo -e "San Francisco 49ers cornerback Shawntae Spencer will miss the rest of the <mask> with a torn ligament in his left knee." | transformers-cli run --task fill-mask --model allenai/longformer-base-4096 --device 0
```

</hfoption>
</hfoptions

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [https://huggingface.co/docs/transformers/en/quantization/bitsandbytes](bitsandbytes) to only quantize the weights to 4-bits.

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = AutoModelForMaskedLM.from_pretrained(
    "allenai/longformer-base-4096",
    quantization_config=bnb_config
    )

text = (
"""
San Francisco 49ers cornerback Shawntae Spencer will miss the rest of the <mask> with a torn ligament in his left knee.
Spencer, a fifth-year pro, will be placed on injured reserve soon after undergoing surgery Wednesday to repair the ligament. He injured his knee late in the 49ers’ road victory at Seattle on Sept. 14, and missed last week’s victory over Detroit.
Tarell Brown and Donald Strickland will compete to replace Spencer with the 49ers, who kept 12 defensive backs on their 53-man roster to start the season. Brown, a second-year pro, got his first career interception last weekend while filling in for Strickland, who also sat out with a knee injury.
"""
)

input_ids = tokenizer([text], return_tensors="pt")["input_ids"]
logits = model(input_ids).logits

masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
probs = logits[0, masked_index].softmax(dim=0)
values, predictions = probs.topk(5)
tokenizer.decode(predictions).split()
```

## Notes

- If you're using Transformers < 4.37.0.dev, set `trust_remote_code=True` in [~AutoModel.from_pretrained]. Otherwise, make sure you update Transformers to the latest stable version.

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = AutoModelForMaskedLM.from_pretrained(
    "allenai/longformer-base-4096",
    trust_remote_code=True,
    )

text = (
"""
San Francisco 49ers cornerback Shawntae Spencer will miss the rest of the <mask> with a torn ligament in his left knee.
Spencer, a fifth-year pro, will be placed on injured reserve soon after undergoing surgery Wednesday to repair the ligament. He injured his knee late in the 49ers’ road victory at Seattle on Sept. 14, and missed last week’s victory over Detroit.
Tarell Brown and Donald Strickland will compete to replace Spencer with the 49ers, who kept 12 defensive backs on their 53-man roster to start the season. Brown, a second-year pro, got his first career interception last weekend while filling in for Strickland, who also sat out with a knee injury.
"""
)

input_ids = tokenizer([text], return_tensors="pt")["input_ids"]
logits = model(input_ids).logits

masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
probs = logits[0, masked_index].softmax(dim=0)
values, predictions = probs.topk(5)
tokenizer.decode(predictions).split()
```

## LongformerConfig

[[autodoc]] LongformerConfig

## LongformerTokenizer

[[autodoc]] LongformerTokenizer

## LongformerTokenizerFast

[[autodoc]] LongformerTokenizerFast

## Longformer specific outputs

[[autodoc]] models.longformer.modeling_longformer.LongformerBaseModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerBaseModelOutputWithPooling

[[autodoc]] models.longformer.modeling_longformer.LongformerMaskedLMOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerSequenceClassifierOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerMultipleChoiceModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerTokenClassifierOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutputWithPooling

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerMaskedLMOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerQuestionAnsweringModelOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerSequenceClassifierOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerMultipleChoiceModelOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerTokenClassifierOutput

## LongformerModel

[[autodoc]] LongformerModel - forward

## LongformerForMaskedLM

[[autodoc]] LongformerForMaskedLM - forward

## LongformerForSequenceClassification

[[autodoc]] LongformerForSequenceClassification - forward

## LongformerForMultipleChoice

[[autodoc]] LongformerForMultipleChoice - forward

## LongformerForTokenClassification

[[autodoc]] LongformerForTokenClassification - forward

## LongformerForQuestionAnswering

[[autodoc]] LongformerForQuestionAnswering - forward

## TFLongformerModel

[[autodoc]] TFLongformerModel - call

## TFLongformerForMaskedLM

[[autodoc]] TFLongformerForMaskedLM - call

## TFLongformerForQuestionAnswering

[[autodoc]] TFLongformerForQuestionAnswering - call

## TFLongformerForSequenceClassification

[[autodoc]] TFLongformerForSequenceClassification - call

## TFLongformerForTokenClassification

[[autodoc]] TFLongformerForTokenClassification - call

## TFLongformerForMultipleChoice

[[autodoc]] TFLongformerForMultipleChoice - call
