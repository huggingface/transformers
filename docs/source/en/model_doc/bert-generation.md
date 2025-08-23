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
    </div>
</div>

# BertGeneration

[BertGeneration](https://huggingface.co/papers/1907.12461) leverages pretrained BERT checkpoints for sequence-to-sequence tasks with the [`EncoderDecoderModel`] architecture. BertGeneration adapts the [`BERT`] for generative tasks.

You can find all the original BERT checkpoints under the [BERT](https://huggingface.co/collections/google/bert-release-64ff5e7a4be99045d1896dbc) collection.

> [!TIP]
> This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).
>
> Click on the BertGeneration models in the right sidebar for more examples of how to apply BertGeneration to different sequence generation tasks.

The example below demonstrates how to use BertGeneration with [`EncoderDecoderModel`] for sequence-to-sequence tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text2text-generation",
    model="google/roberta2roberta_L-24_discofuse",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create energy through ")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import EncoderDecoderModel, AutoTokenizer

model = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

input_ids = tokenizer(
    "Plants create energy through ", add_special_tokens=False, return_tensors="pt"
).input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through " | transformers run --task text2text-generation --model "google/roberta2roberta_L-24_discofuse" --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [BitsAndBytesConfig](../quantizationbitsandbytes) to quantize the weights to 4-bit.

```python
import torch
from transformers import EncoderDecoderModel, AutoTokenizer, BitsAndBytesConfig

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = EncoderDecoderModel.from_pretrained(
    "google/roberta2roberta_L-24_discofuse",
    quantization_config=quantization_config,
    dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

input_ids = tokenizer(
    "Plants create energy through ", add_special_tokens=False, return_tensors="pt"
).input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

## Notes

- [`BertGenerationEncoder`] and [`BertGenerationDecoder`] should be used in combination with [`EncoderDecoderModel`] for sequence-to-sequence tasks.

   ```python
   from transformers import BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, EncoderDecoderModel
   
   # leverage checkpoints for Bert2Bert model
   # use BERT's cls token as BOS token and sep token as EOS token
   encoder = BertGenerationEncoder.from_pretrained("google-bert/bert-large-uncased", bos_token_id=101, eos_token_id=102)
   # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
   decoder = BertGenerationDecoder.from_pretrained(
       "google-bert/bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
   )
   bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

   # create tokenizer
   tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")

   input_ids = tokenizer(
       "This is a long article to summarize", add_special_tokens=False, return_tensors="pt"
   ).input_ids
   labels = tokenizer("This is a short summary", return_tensors="pt").input_ids

   # train
   loss = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels).loss
   loss.backward()
   ```

- For summarization, sentence splitting, sentence fusion and translation, no special tokens are required for the input.
- No EOS token should be added to the end of the input for most generation tasks.

## BertGenerationConfig

[[autodoc]] BertGenerationConfig

## BertGenerationTokenizer

[[autodoc]] BertGenerationTokenizer
    - save_vocabulary

## BertGenerationEncoder

[[autodoc]] BertGenerationEncoder
    - forward

## BertGenerationDecoder

[[autodoc]] BertGenerationDecoder
    - forward