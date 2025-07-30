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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# BigBirdPegasus

[BigBirdPegasus](https://huggingface.co/papers/2007.14062) is an encoder-decoder (sequence-to-sequence) transformer model for long-input summarization. It extends the [BigBird](./big_bird) architecture with an additional pretraining objective borrowed from [Pegasus](./pegasus) called gap sequence generation (GSG). Whole sentences are masked and the model has to fill in the gaps in the document. BigBirdPegasus's ability to keep track of long contexts makes it effective at summarizing lengthy inputs, surpassing the performance of base Pegasus models.

You can find all the original BigBirdPegasus checkpoints under the [Google](https://huggingface.co/google/models?search=bigbird-pegasus) organization.

> [!TIP]
> This model was contributed by [vasudevgupta](https://huggingface.co/vasudevgupta).
>
> Click on the BigBirdPegasus models in the right sidebar for more examples of how to apply BigBirdPegasus to different language tasks.

The example below demonstrates how to summarize text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="summarization",
    model="google/bigbird-pegasus-large-arxiv",
    dtype=torch.float32,
    device=0
)
pipeline("""Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle.""")
```
</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    "google/bigbird-pegasus-large-arxiv"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/bigbird-pegasus-large-arxiv",
    dtype=torch.bfloat16,
    device_map="auto",
)

input_text = """Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle."""
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>
<hfoption id="transformers-cli">

```bash
echo -e "Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet. Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts." | transformers-cli run --task summarization --model google/bigbird-pegasus-large-arxiv --device 0
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
    "google/bigbird-pegasus-large-arxiv",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(
    "google/bigbird-pegasus-large-arxiv"
)

input_text = """Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle."""
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Notes

- BigBirdPegasus also uses the [`PegasusTokenizer`].
- Inputs should be padded on the right because BigBird uses absolute position embeddings.
- BigBirdPegasus supports `original_full` and `block_sparse` attention. If the input sequence length is less than 1024, it is recommended to use `original_full` since sparse patterns don't offer much benefit for smaller inputs.
- The current implementation uses window size of 3 blocks and 2 global blocks, only supports the ITC-implementation, and doesn't support `num_random_blocks=0`.
- The sequence length must be divisible by the block size.

## Resources

Read the [Understanding BigBird's Block Sparse Attention](https://huggingface.co/blog/big-bird) blog post for more details about how BigBird's attention works.

## BigBirdPegasusConfig

[[autodoc]] BigBirdPegasusConfig
    - all

## BigBirdPegasusModel

[[autodoc]] BigBirdPegasusModel
    - forward

## BigBirdPegasusForConditionalGeneration

[[autodoc]] BigBirdPegasusForConditionalGeneration
    - forward

## BigBirdPegasusForSequenceClassification

[[autodoc]] BigBirdPegasusForSequenceClassification
    - forward

## BigBirdPegasusForQuestionAnswering

[[autodoc]] BigBirdPegasusForQuestionAnswering
    - forward

## BigBirdPegasusForCausalLM

[[autodoc]] BigBirdPegasusForCausalLM
    - forward
