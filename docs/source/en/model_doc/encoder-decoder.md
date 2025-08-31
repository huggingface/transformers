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
*This model was released on 2017-06-12 and added to Hugging Face Transformers on 2020-11-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Encoder Decoder Models

[`EncoderDecoderModel`](https://huggingface.co/papers/1706.03762) initializes a sequence-to-sequence model with any pretrained autoencoder and pretrained autoregressive model. It is effective for sequence generation tasks as demonstrated in [Text Summarization with Pretrained Encoders](https://huggingface.co/papers/1908.08345) which uses [`BertModel`] as the encoder and decoder.

> [!TIP]
> This model was contributed by [thomwolf](https://huggingface.co/thomwolf).
>
> Click on the Encoder Decoder models in the right sidebar for more examples of how to apply Encoder Decoder to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="patrickvonplaten/bert2bert-cnn_dailymail-fp16",
    device=0
)

text = "Plants create energy through a process known as photosynthesis. This involves capturing sunlight and converting carbon dioxide and water into glucose and oxygen."
print(summarizer(text))
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")
model = AutoModelForCausalLM.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16", dtype=torch.bfloat16, device_map="auto",attn_implementation="sdpa")

text = "Plants create energy through a process known as photosynthesis. This involves capturing sunlight and converting carbon dioxide and water into glucose and oxygen."

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)

summary = model.generate(**inputs, max_length=60, num_beams=4, early_stopping=True)
print(tokenizer.decode(summary[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as photosynthesis. This involves capturing sunlight and converting carbon dioxide and water into glucose and oxygen." | transformers-cli run --task summarization --model "patrickvonplaten/bert2bert-cnn_dailymail-fp16" --device 0
```

</hfoption>
</hfoptions>

## Notes

- [`EncoderDecoderModel`] can be initialized using any pretrained encoder and decoder. But depending on the decoder architecture, the cross-attention layers may be randomly initialized.

These models require downstream fine-tuning, as discussed in this [blog post](https://huggingface.co/blog/warm-starting-encoder-decoder). Use [`~EncoderDecoderModel.from_encoder_decoder_pretrained`] to combine encoder and decoder checkpoints.

```python
from transformers import EncoderDecoderModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "google-bert/bert-base-uncased",
    "google-bert/bert-base-uncased"
)
```

- Encoder Decoder models can be fine-tuned like BART, T5 or any other encoder-decoder model. Only 2 inputs are required to compute a loss, `input_ids` and `labels`. Refer to this [notebook](https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=ZwQIEhKOrJpl) for a more detailed training example.

```python
>>> from transformers import BertTokenizer, EncoderDecoderModel

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased", "google-bert/bert-base-uncased")

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> input_ids = tokenizer(
...     "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
...     return_tensors="pt",
... ).input_ids

>>> labels = tokenizer(
...     "the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris.",
...     return_tensors="pt",
... ).input_ids

>>> # the forward function automatically creates the correct decoder_input_ids
>>> loss = model(input_ids=input_ids, labels=labels).loss
```

- [`EncoderDecoderModel`] can be randomly initialized from an encoder and a decoder config as shown below.

```python
>>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

>>> config_encoder = BertConfig()
>>> config_decoder = BertConfig()

>>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = EncoderDecoderModel(config=config)
```

- The Encoder Decoder Model can also be used for translation as shown below.

```python
from transformers import AutoTokenizer, EncoderDecoderModel

# Load a pre-trained translation model
model_name = "google/bert2bert_L-24_wmt_en_de"
tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="<pad>", eos_token="</s>", bos_token="<s>")
model = EncoderDecoderModel.from_pretrained(model_name)

# Input sentence to translate
input_text = "Plants create energy through a process known as"

# Encode the input text
inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).input_ids

# Generate the translated output
outputs = model.generate(inputs)[0]

# Decode the output tokens to get the translated sentence
translated_text = tokenizer.decode(outputs, skip_special_tokens=True)

print("Translated text:", translated_text)
```

## EncoderDecoderConfig

[[autodoc]] EncoderDecoderConfig

## EncoderDecoderModel

[[autodoc]] EncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained
