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
*This model was released on 2019-10-29 and added to Hugging Face Transformers on 2020-11-16.*


<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# BART
[BART](https://huggingface.co/papers/1910.13461) is a sequence-to-sequence model that combines the pretraining objectives from BERT and GPT. It’s pretrained by corrupting text in different ways like deleting words, shuffling sentences, or masking tokens and learning how to fix it. The encoder encodes the corrupted document and the corrupted text is fixed by the decoder. As it learns to recover the original text, BART gets really good at both understanding and generating language.

You can find all the original BART checkpoints under the [AI at Meta](https://huggingface.co/facebook?search_models=bart) organization.

The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="facebook/bart-large",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create <mask> through a process known as photosynthesis.")

```
</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-large",
)
model = AutoModelForMaskedLM.from_pretrained(
    "facebook/bart-large",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create <mask> through a process known as photosynthesis." | transformers-cli run --task fill-mask --model facebook/bart-large --device 0
```

</hfoption>
</hfoptions>

## Notes

- Inputs should be padded on the right because BERT uses absolute position embeddings.
- The [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) checkpoint doesn't include `mask_token_id` which means it can't perform mask-filling tasks.
- BART doesn’t use `token_type_ids` for sequence classification. Use [`BartTokenizer`] or [`~PreTrainedTokenizerBase.encode`] to get the proper splitting.
- The forward pass of [`BartModel`] creates the `decoder_input_ids` if they're not passed. This can be different from other model APIs, but it is a useful feature for mask-filling tasks.
- Model predictions are intended to be identical to the original implementation when `forced_bos_token_id=0`. This only works if the text passed to `fairseq.encode` begins with a space.
- [`~GenerationMixin.generate`] should be used for conditional generation tasks like summarization.

## BartConfig

[[autodoc]] BartConfig
    - all

## BartTokenizer

[[autodoc]] BartTokenizer
    - all

## BartTokenizerFast

[[autodoc]] BartTokenizerFast
    - all

## BartModel

[[autodoc]] BartModel
    - forward

## BartForConditionalGeneration

[[autodoc]] BartForConditionalGeneration
    - forward

## BartForSequenceClassification

[[autodoc]] BartForSequenceClassification
    - forward

## BartForQuestionAnswering

[[autodoc]] BartForQuestionAnswering
    - forward

## BartForCausalLM

[[autodoc]] BartForCausalLM
    - forward
