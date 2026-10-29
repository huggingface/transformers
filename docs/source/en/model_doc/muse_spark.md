<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-14.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Muse Spark Open

## Overview

Muse Spark Open is a multimodal mixture-of-experts (MoE) model developed by Meta that accepts text, images, audio, and video as input and generates text. It has 816 billion total parameters, with 42 billion active per token, and supports a context window of up to 1,048,576 tokens shared across all input modalities. The model is trained for agentic coding tasks, including multi-file code editing, debugging, and tool use.

The language backbone consists of 62 transformer layers, each routing tokens to 8 of 256 experts. It uses grouped-query attention with query-key normalization, attention output gating and normalization, and learned residual gates. A shared perception encoder processes images and video frames, while a convolutional audio encoder produces audio representations at 12.5 tokens per second. Muse Spark Open shares its tokenizer and vocabulary with [Muse Glimmer](./muse_glimmer), with extensions to the chat template for multimodal inputs. Checkpoints are available in BF16 and [NVFP4 formats](#nvfp4-weights).

<div class="flex justify-center">
   <img src = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/muse_spark.jpeg">
</div>


## Resources

- [Hugging Face Collection](https://huggingface.co/collections/meta-models/muse-spark-open)
- [Hugging Face Blog](http://hf.co/blog/meta-muse-spark-open)
- [Post-train with TRL](https://github.com/huggingface/trl/tree/main/examples/sft_muse_spark_open)

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]), # Number of devices across which the dense layers are sharded
    ep_size=int(os.environ["WORLD_SIZE"]), # Number of devices across which the experts are distributed
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B",
    distributed_config=distributed_config,
)

messages = [
    {"role": "user", "content": "Explain in one sentence why the sky is usually blue."},
]
inputs = processor.apply_chat_template(
    messages,
    reasoning_effort="low",  # default: xhigh, options: low/medium/high/xhigh
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
#  to=self<|message|>Explain in one sentence why the sky is usually blue.
# ...
# <|eom|><|start|>assistant to=user<|message|>The sky is usually blue because molecules in Earth's atmosphere scatter shorter-wavelength blue sunlight in all directions much more strongly than longer-wavelength red light.<|eot|>
```

</hfoption>
<hfoption id="AutoModel Image">

```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B",
    distributed_config=distributed_config,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    reasoning_effort="medium", # default: xhigh, options: low/medium/high/xhigh
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(response)
#  to=self<|message|>What is shown in this image?
# ...
# <|eom|><|start|>assistant to=user<|message|>The image shows a **Pallas's cat — also known as a manul, *Otocolobus manul*** — walking through snow.
#
# It is a small, stocky wild cat recognizable by:
#
# * **Its extremely dense, long, greyish-brown winter coat** with faint dark stripes and a dusting of snow on its back
# * **Its flat, broad face** with low-set ears, a short muzzle, and dark stripes on the cheeks
# * **Its short legs and thick, plump body** adapted for cold climates
#
# The cat is captured mid-stride on a snow-covered surface, with one front paw lifted. In the background are white birch tree trunks with black markings and a chain-link fence on the left, indicating the animal is in a zoo or wildlife enclosure in winter.<|eot|>
```

</hfoption>
<hfoption id="AutoModel Video">

```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B",
    distributed_config=distributed_config,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"},
            {"type": "text", "text": "What is happening in this video?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    reasoning_effort="medium",  # default: xhigh, options: low/medium/high/xhigh
    processor_kwargs={
        "fps": 2,   # default: 2, number of frames sampled per second
        "num_frames": 8, # default: 94, maximum number of frames sampled from the video
    },
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
#  to=self<|message|>What is happening in this video?
# ...
# <|eom|><|start|>assistant to=user<|message|>In this video two people are playing tennis on an indoor court.
#
# The shot is taken from behind one of the players - a man in a white t-shirt, black shorts and white shoes and socks. He is standing near the baseline on an orange hard court with white lines, in a ready position with his knees bent and his racket held in front of him.
#
# Across the net, on the far side of the court, is his opponent, also dressed in light-colored clothing.
#
# The sequence shows a short rally:
#
# *   The opponent hits the ball over the net.
# *   The man in the foreground shuffles to his left, turns his body, and prepares for a shot.
# *   He executes a two-handed backhand, swinging the racket across his body and following through as he hits the ball back over the net.
#
# The setting is a large indoor tennis hall with a high, arched ceiling with exposed wooden beams and overhead fluorescent lights. The court is surrounded by a grey run-off area, green windscreening on the walls, and netting that separates it from adjacent courts. On the right side you can see an umpire's chair and a few yellow and black benches. An "Adobe Stock" watermark is visible in the center of the frame.<|eot|>
```

</hfoption>
<hfoption id="AutoModel Audio">

```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B",
    distributed_config=distributed_config,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/glass_breaking.mp3"},
            {"type": "text", "text": "What is said in this recording?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    reasoning_effort="low", # default: xhigh, options: low/medium/high/xhigh
    processor_kwargs={"sampling_rate": 24000},  # sampling rate the model was trained with
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
#  to=self<|message|>We need to respond to user request: "What is said in this recording?"
# ...
# <|eom|><|start|>assistant to=user<|message|>In the recording the person says:
#
# > "Just picked up this glass and the people at the store told me that if I dropped it it wouldn't break."
#
# Then the glass is dropped and shatters.<|eot|>
```

</hfoption>
</hfoptions>

## NVFP4 Weights

NVFP4 [quantized weights](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) with lower memory requirements are available on the hub. The model can be loaded on any NVFP4 compatible device without changes to the code. The quantization configuration is automatically applied at load time.

> [!IMPORTANT]
> For NVFP4 support, transformers must be installed from [this branch](https://github.com/huggingface/transformers/pull/49015).
>
> ```bash
> pip install git+https://github.com/huggingface/transformers.git@finegrained-per-scheme-demo
> ```


```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B-NVFP4")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    distributed_config=distributed_config,
)

# ...
```

## MuseSparkConfig

[[autodoc]] MuseSparkConfig

## MuseSparkTextConfig

[[autodoc]] MuseSparkTextConfig

## MuseSparkVisionConfig

[[autodoc]] MuseSparkVisionConfig

## MuseSparkAudioConfig

[[autodoc]] MuseSparkAudioConfig

## MuseSparkFeatureExtractor

[[autodoc]] MuseSparkFeatureExtractor

## MuseSparkProcessor

[[autodoc]] MuseSparkProcessor

## MuseSparkPreTrainedModel

[[autodoc]] MuseSparkPreTrainedModel

## MuseSparkTextModel

[[autodoc]] MuseSparkTextModel
    - forward

## MuseSparkVisionModel

[[autodoc]] MuseSparkVisionModel
    - forward

## MuseSparkAudioModel

[[autodoc]] MuseSparkAudioModel
    - forward

## MuseSparkModel

[[autodoc]] MuseSparkModel
    - forward

## MuseSparkForConditionalGeneration

[[autodoc]] MuseSparkForConditionalGeneration
    - forward
