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
            <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    </div>
</div>

# LED

[Longformer-Encoder-Decoder (LED)](https://huggingface.co/papers/2004.05150) is an encoder-decoder  transformer model for sequence-to-sequence tasks like summarization. It extends [Longformer](.longformer), an encoder-only model designed to handle long inputs, by adding a decoder layer. The decoder uses full self-attention on the encoded tokens and previously decoded locations. Because of Longformer's linear self-attention mechanism, LED is more efficient than standard encoder-decoder models when processing long sequences.

You can find all the original [LED] checkpoints under the [Ai2](https://huggingface.co/allenai/models?search=led) organization.

> [!TIP]
> This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).
>
> Click on the LED models in the right sidebar for more examples of how to apply LED to different language tasks.

The example below demonstrates how to summarize text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="summarization",
    model="allenai/led-base-16384",
    dtype=torch.float16,
    device=0
)
pipeline("""Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle.""")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    "allenai/led-base-16384"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "allenai/led-base-16384",
    dtype=torch.float16,
    device_map="auto"
)

input_text = """Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle."""
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# Place global attention on the first token
global_attention_mask = torch.zeros_like(input_ids.input_ids).to("cuda")
global_attention_mask[:, 0] = 1

output = model.generate(**input_ids, global_attention_mask=global_attention_mask, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers-cli">

```bash
!echo -e "Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet. Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts." | transformers-cli run --task summarization --model allenai/led-base-16384 --device 0
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
    "allenai/led-large-16384",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(
    "allenai/led-large-16384"
)

input_text = """Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle."""
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# Place global attention on the first token
global_attention_mask = torch.zeros_like(input_ids.input_ids).to("cuda")
global_attention_mask[:, 0] = 1

output = model.generate(**input_ids, global_attention_mask=global_attention_mask, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Notes

- [`LEDForConditionalGeneration`] is an extension of [`BartForConditionalGeneration`] exchanging the traditional self-attention layer with Longformer's chunked self-attention layer. [`LEDTokenizer`] is an alias of [`BartTokenizer`].
- LED pads the `input_ids` to be a multiple of `config.attention_window` if required. A small speedup is gained when [`LEDTokenizer`] is used with the `pad_to_multiple_of` argument.
- LED works best on long-range sequence-to-sequence tasks where the `input_ids` are significantly longer than 1024 tokens.
- LED uses global attention by means of the `global_attention_mask` (see [`LongformerModel`]). For summarization, it is advised to put global attention only on the first `<s>` token. For question answering, it is advised to put global attention on all tokens of the question.
- To fine-tune LED on all 16384 parameters, gradient checkpointing can be enabled in case training leads to out-of-memory (OOM) errors. Enable gradient checkpointing by adding `model.gradient_checkpointing_enable()` and setting `use_cache=False` to disable the caching mechanism to save memory.
- Inputs should be padded on the right because LED uses absolute position embeddings.

## Resources

- Read the [LED on Arxiv notebook](https://colab.research.google.com/drive/12INTTR6n64TzS4RrXZxMSXfrOd9Xzamo?usp=sharing) to see how LED can achieve state-of-the-art performance on Arxiv article summarization.
- Read the [Fine-tune LED notebook](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing) to learn how to fine-tune LED on PubMed articles.

## LEDConfig

[[autodoc]] LEDConfig

## LEDTokenizer

[[autodoc]] LEDTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LEDTokenizerFast

[[autodoc]] LEDTokenizerFast

## LED specific outputs

[[autodoc]] models.led.modeling_led.LEDEncoderBaseModelOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqModelOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqLMOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput

[[autodoc]] models.led.modeling_tf_led.TFLEDEncoderBaseModelOutput

[[autodoc]] models.led.modeling_tf_led.TFLEDSeq2SeqModelOutput

[[autodoc]] models.led.modeling_tf_led.TFLEDSeq2SeqLMOutput

<frameworkcontent>
<pt>

## LEDModel

[[autodoc]] LEDModel
    - forward

## LEDForConditionalGeneration

[[autodoc]] LEDForConditionalGeneration
    - forward

## LEDForSequenceClassification

[[autodoc]] LEDForSequenceClassification
    - forward

## LEDForQuestionAnswering

[[autodoc]] LEDForQuestionAnswering
    - forward

</pt>
<tf>

## TFLEDModel

[[autodoc]] TFLEDModel
    - call

## TFLEDForConditionalGeneration

[[autodoc]] TFLEDForConditionalGeneration
    - call

</tf>
</frameworkcontent>



