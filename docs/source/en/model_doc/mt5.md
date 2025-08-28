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
*This model was released on 2020-10-22 and added to Hugging Face Transformers on 2020-11-17.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# mT5

[mT5](https://huggingface.co/papers/2010.11934) is a multilingual variant of [T5](./t5), training on 101 languages. It also incorporates a new "accidental translation" technique to prevent the model from incorrectly translating predictions into the wrong language.

You can find all the original [mT5] checkpoints under the [mT5](https://huggingface.co/collections/google/mt5-release-65005f1a520f8d7b4d039509) collection.

> [!TIP]
> This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).
>
> Click on the mT5 models in the right sidebar for more examples of how to apply mT5 to different language tasks.

The example below demonstrates how to summarize text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text2text-generation",
    model="csebuetnlp/mT5_multilingual_XLSum",
    dtype=torch.float16,
    device=0
)
pipeline("""Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.""")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "csebuetnlp/mT5_multilingual_XLSum"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "csebuetnlp/mT5_multilingual_XLSum",
    dtype=torch.float16,
    device_map="auto",
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
echo -e "Plants are remarkable organisms that produce their own food using a method called photosynthesis." | transformers run --task text2text-generation --model csebuetnlp/mT5_multilingual_XLSum --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to int4.

```python
import torch
from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoTokenizer

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "csebuetnlp/mT5_multilingual_XLSum",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(
    "csebuetnlp/mT5_multilingual_XLSum"
)
input_text = """Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems."""
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Notes

- mT5 must be fine-tuned for downstream tasks because it was only pretrained on the [mc4](https://huggingface.co/datasets/mc4) dataset.

## MT5Config

[[autodoc]] MT5Config

## MT5Tokenizer

[[autodoc]] MT5Tokenizer

See [`T5Tokenizer`] for all details.


## MT5TokenizerFast

[[autodoc]] MT5TokenizerFast

See [`T5TokenizerFast`] for all details.

## MT5Model

[[autodoc]] MT5Model

## MT5ForConditionalGeneration

[[autodoc]] MT5ForConditionalGeneration

## MT5EncoderModel

[[autodoc]] MT5EncoderModel

## MT5ForSequenceClassification

[[autodoc]] MT5ForSequenceClassification

## MT5ForTokenClassification

[[autodoc]] MT5ForTokenClassification

## MT5ForQuestionAnswering

[[autodoc]] MT5ForQuestionAnswering
