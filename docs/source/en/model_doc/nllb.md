<!--Copyright 2020 The HuggingFace Team. All rights reserved.

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
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

*This model was released on 2022-07-11 and added to Hugging Face Transformers on 2022-07-18.*


# NLLB

[NLLB: No Language Left Behind](https://huggingface.co/papers/2207.04672) is a multilingual translation model. It's trained on data using data mining techniques tailored for low-resource languages and supports over 200 languages. NLLB features a conditional compute architecture using a Sparsely Gated Mixture of Experts.


You can find all the original NLLB checkpoints under the [AI at Meta](https://huggingface.co/facebook/models?search=nllb) organization.

> [!TIP]
> This model was contributed by [Lysandre](https://huggingface.co/lysandre).  
> Click on the NLLB models in the right sidebar for more examples of how to apply NLLB to different translation tasks.

The example below demonstrates how to translate text with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(task="translation", model="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn", dtype=torch.float16, device=0)
pipeline("UN Chief says there is no military solution in Syria")
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", dtype="auto", attn_implementaiton="sdpa")

article = "UN Chief says there is no military solution in Syria"
inputs = tokenizer(article, return_tensors="pt")

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"), max_length=30
)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "UN Chief says there is no military solution in Syria" | transformers run --task "translation_en_to_fr" --model facebook/nllb-200-distilled-600M --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 8-bits.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")

article = "UN Chief says there is no military solution in Syria"
inputs = tokenizer(article, return_tensors="pt").to(model.device)
translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"), max_length=30,
)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```python
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("facebook/nllb-200-distilled-600M")
visualizer("UN Chief says there is no military solution in Syria")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/NLLB-Attn-Mask.png"/>
</div>

## Notes

- The tokenizer was updated in April 2023 to prefix the source sequence with the source language rather than the target language. This prioritizes zero-shot performance at a minor cost to supervised performance.

   ```python
   >>> from transformers import NllbTokenizer

   >>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
   >>> tokenizer("How was your day?").input_ids
   [256047, 13374, 1398, 4260, 4039, 248130, 2]
   ```
   
   To revert to the legacy behavior, use the code example below.
   
   ```python
   >>> from transformers import NllbTokenizer

   >>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
   ```
   
 - For non-English languages, specify the language's [BCP-47](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) code with the `src_lang` keyword as shown below.
 
 - See example below for a translation from Romanian to German.
    ```python
    >>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    >>> article = "UN Chief says there is no military solution in Syria"
    >>> inputs = tokenizer(article, return_tensors="pt")

    >>> translated_tokens = model.generate(
    ...     **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"), max_length=30
    ... )
    >>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    Le chef de l'ONU dit qu'il n'y a pas de solution militaire en Syrie
    ```

## NllbTokenizer

[[autodoc]] NllbTokenizer
    - build_inputs_with_special_tokens

## NllbTokenizerFast

[[autodoc]] NllbTokenizerFast
