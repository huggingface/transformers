<!--Copyright 2026 MiniMaxAI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-03.*

# MiniMax-VL-01

## Overview

[MiniMax-VL-01](https://huggingface.co/MiniMaxAI/MiniMax-VL-01) is the vision-language model introduced in
[MiniMax-01: Scaling Foundation Models with Lightning Attention](https://huggingface.co/papers/2501.08313) by MiniMax.
It accepts images and text as input and generates text. The model combines three components:

- A 24-layer CLIP-style vision encoder, trained from scratch, processes 336×336 image crops with a patch size of 14.
- A two-layer projector with a GELU activation maps vision features to the language model's hidden size.
- The 456B-parameter MiniMax-Text-01 backbone has 80 decoder layers. Each group of eight layers contains seven
  Lightning Attention layers followed by one full-attention layer. Each layer has 32 experts and selects two per token.

Images use an any-resolution layout: a 336×336 thumbnail followed by a grid of 336×336 crops selected for the image's
aspect ratio. The projected crop features are reassembled, unpadded, and separated by learned newline embeddings.
These features replace the expanded `<image>` tokens in the text prompt.

This model was contributed by [sushmanth](https://huggingface.co/sushmanth).
The original code can be found in the [MiniMax-VL-01 checkpoint repository](https://huggingface.co/MiniMaxAI/MiniMax-VL-01/tree/main).

## Usage

### Prepare the configuration

The released checkpoint uses a custom configuration schema. From the root of the Transformers repository, download
its configuration and convert it to the native [`MiniMaxVL01Config`] format:

```bash
hf download MiniMaxAI/MiniMax-VL-01 config.json --local-dir ./MiniMax-VL-01

python src/transformers/models/minimax_vl_01/convert_minimax_vl_01_weights_to_hf.py \
    ./MiniMax-VL-01 ./MiniMax-VL-01-native-config
```

If you already have the original `config.json` locally, use its directory as the converter's first argument.
The converter writes only a new `config.json`; the output directory is not a complete checkpoint. The example below
loads the converted configuration locally and the weights, tokenizer, and image processor from the original checkpoint.
Pass the converted configuration to both the model and the processor when loading.
Transformers applies the weight-name and expert-layout conversions during model loading.

### Image-text generation

Use [`MiniMaxVL01Processor`] to format the conversation and prepare images. Its chat template supplies the checkpoint's
role markers, and the processor expands each image placeholder to match that image's visual features.

> [!NOTE]
> The full checkpoint occupies approximately 913 GB in 16-bit precision. The example uses [`DistributedConfig`] for
> tensor and expert parallelism. Use GPUs with enough memory for the model and inference, and PyTorch 2.7 or later
> for expert parallelism.

Save the example below as `minimax_vl_01_inference.py`. Set `NUM_GPUS` to the number of GPUs to use, which must
evenly divide the checkpoint's 8 key-value heads and 32 experts, and launch one process per GPU:

```bash
torchrun --nproc_per_node="$NUM_GPUS" minimax_vl_01_inference.py
```

```python
import os

import torch

from transformers import AutoModelForImageTextToText, DistributedConfig, MiniMaxVL01Config, MiniMaxVL01Processor


checkpoint = "MiniMaxAI/MiniMax-VL-01"
world_size = int(os.environ["WORLD_SIZE"])
config = MiniMaxVL01Config.from_pretrained("./MiniMax-VL-01-native-config")
processor = MiniMaxVL01Processor.from_pretrained(checkpoint, config=config)
model = AutoModelForImageTextToText.from_pretrained(
    checkpoint,
    config=config,
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=world_size, ep_size=world_size),
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
            },
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False,
    eos_token_id=processor.tokenizer.eos_token_id,
    pad_token_id=processor.tokenizer.eos_token_id,
)

if int(os.environ["RANK"]) == 0:
    response_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    print(processor.batch_decode(response_ids, skip_special_tokens=True)[0])
```

## MiniMaxVL01Config

[[autodoc]] MiniMaxVL01Config

## MiniMaxVL01TextConfig

[[autodoc]] MiniMaxVL01TextConfig

## MiniMaxVL01ImageProcessor

[[autodoc]] MiniMaxVL01ImageProcessor
    - preprocess

## MiniMaxVL01ImageProcessorPil

[[autodoc]] MiniMaxVL01ImageProcessorPil
    - preprocess

## MiniMaxVL01Processor

[[autodoc]] MiniMaxVL01Processor
    - __call__

## MiniMaxVL01Model

[[autodoc]] MiniMaxVL01Model
    - forward
    - get_image_features

## MiniMaxVL01TextModel

[[autodoc]] MiniMaxVL01TextModel
    - forward

## MiniMaxVL01ForConditionalGeneration

[[autodoc]] MiniMaxVL01ForConditionalGeneration
    - forward
    - get_image_features
