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
*This model was released on 2019-12-18 and added to Hugging Face Transformers on 2020-11-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Pegasus

[Pegasus](https://huggingface.co/papers/1912.08777) is an encoder-decoder (sequence-to-sequence) transformer model pretrained on unlabeled text to perform abstractive summarization. Pegasus is trained jointly on two self-supervised objective functions, masked language modeling (MLM) and gap sentence generation (GSG). Whole sentences are masked and the model has to fill in the gaps in the document. It can be fine-tuned with good performance even on small datasets with only 1000 examples.

You can find all the original Pegasus checkpoints under the [Google](https://huggingface.co/google?search_models=pegasus) organization.

> [!TIP]
> Click on the Pegasus models in the right sidebar for more examples of how to apply Pegasus to different language tasks.

The example below demonstrates how to summarize text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="summarization",
    model="google/pegasus-xsum",
    dtype=torch.float16,
    device=0
)
pipeline("""Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.""")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/pegasus-xsum"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/pegasus-xsum",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

input_text = """Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems."""
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants are remarkable organisms that produce their own food using a method called photosynthesis. This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth. Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems." | transformers-cli run --task summarization --model google/pegasus-xsum --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to int4.

```py
import torch
from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoTokenizer

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/pegasus-xsum",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(
    "google/pegasus-xsum"
)
input_text = """Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems."""
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Notes

- [`AdaFactor`] is the recommended optimizer for fine-tuning Pegasus.
- This implementation of Pegasus inherits from [`BartForConditionalGeneration`] but it uses static/sinusoidal positional embeddings instead. Pegasus also starts generating with `pad_token_id` as the prefix and uses `num_beams=8`.

## PegasusConfig

[[autodoc]] PegasusConfig

## PegasusTokenizer

warning: `add_tokens` does not work at the moment.

[[autodoc]] PegasusTokenizer

## PegasusTokenizerFast

[[autodoc]] PegasusTokenizerFast

## PegasusModel

[[autodoc]] PegasusModel
    - forward

## PegasusForConditionalGeneration

[[autodoc]] PegasusForConditionalGeneration
    - forward

## PegasusForCausalLM

[[autodoc]] PegasusForCausalLM
    - forward
