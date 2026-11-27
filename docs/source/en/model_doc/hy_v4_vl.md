<!--Copyright 2026 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was contributed to Hugging Face Transformers on 2026-11-17.*

# HYV4VL

## Overview

HYV4VL combines the [HYV4](./hy_v4) mixture-of-experts text backbone with a vision encoder for
image-text and video-text generation. It supports dynamic-resolution visual inputs and uses
multimodal rotary position embeddings (mRoPE) to represent temporal and spatial positions.

## Usage

The examples use the tentative model ID `tencent/Hy4`, which may change before release.
The full model requires multiple GPUs. The examples shard the dense layers with tensor parallelism and
distribute the experts with expert parallelism, which is more efficient for this mixture-of-experts model
than tensor parallelism alone.

Save either example as `generate.py` and launch it with one process per GPU. For a single-node
allocation, set `NUM_GPUS` to the number of GPUs to use and run:

```bash
torchrun --standalone --nproc-per-node="$NUM_GPUS" generate.py
```

For multi-node allocations, configure your distributed launcher's node count, node rank, and
rendezvous address. Model loading initializes distributed execution and selects each rank's device.

### Image inputs

```python
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, DistributedConfig

model_id = "tencent/Hy4"
processor = AutoProcessor.from_pretrained(model_id)

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),  # Number of devices across which the dense layers are sharded
    ep_size=int(os.environ["WORLD_SIZE"]),  # Number of devices across which the experts are distributed
)
model = AutoModelForImageTextToText.from_pretrained(model_id, distributed_config=distributed_config)

messages = [{
    "role": "user",
    "content": [
        {
            "type": "image",
            "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        },
        {"type": "text", "text": "Describe this image in one sentence."},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    reasoning_effort="no_think",
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={"images_kwargs": {"device": model.device}},
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

if torch.distributed.get_rank() == 0:
    completion = outputs[0, inputs["input_ids"].shape[1]:]

    print(processor.decode(completion, skip_special_tokens=True))
```

For text-only generation, use a message containing only a `"text"` item.

### Video inputs

Video-file decoding requires TorchCodec and its FFmpeg runtime dependencies. This example
samples four frames; set `fps=None` when providing `num_frames` to disable the default frame rate.

```python
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, DistributedConfig

model_id = "tencent/Hy4"
processor = AutoProcessor.from_pretrained(model_id)

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),  # Number of devices across which the dense layers are sharded
    ep_size=int(os.environ["WORLD_SIZE"]),  # Number of devices across which the experts are distributed
)
model = AutoModelForImageTextToText.from_pretrained(model_id, distributed_config=distributed_config)

messages = [{
    "role": "user",
    "content": [
        {
            "type": "video",
            "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
        },
        {"type": "text", "text": "Summarize the action in this video."},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    reasoning_effort="no_think",
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={
        "videos_kwargs": {
            "do_sample_frames": True,
            "num_frames": 4,
            "fps": None,
            "device": model.device,
        },
    },
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

if torch.distributed.get_rank() == 0:
    completion = outputs[0, inputs["input_ids"].shape[1]:]

    print(processor.decode(completion, skip_special_tokens=True))
```

## Usage Tips

- The chat template defaults to `reasoning_effort="high"`. Use `reasoning_effort="no_think"` for
  a direct answer. For reasoning, allow enough generated tokens for both reasoning and the final answer.
- Load the processor from the checkpoint to use its image and video settings. Images use
  [`HunYuanVLImageProcessor`]; video preprocessing uses [`HYV4VLVideoProcessor`].
- Pass image and video overrides separately through `processor_kwargs["images_kwargs"]` and
  `processor_kwargs["videos_kwargs"]`. Image `size["longest_edge"]` is a per-image pixel budget;
  video `size["longest_edge"]` is a whole-clip pixel budget, so sampling more frames can reduce
  each frame's resolution. Video `max_image_size["longest_edge"]` additionally caps each frame's
  pixel area, not its edge length.
- Video sampling is disabled by default. Enable `do_sample_frames` and pass either `num_frames`
  or `fps`; frame-rate sampling requires source frame-rate metadata.
- For batches with different text lengths, pass `padding=True` in `processor_kwargs["text_kwargs"]`.

## HYV4VLConfig

[[autodoc]] HYV4VLConfig

## HYV4VLTextConfig

[[autodoc]] HYV4VLTextConfig

## HYV4VLVisionConfig

[[autodoc]] HYV4VLVisionConfig

## HYV4VLProcessor

[[autodoc]] HYV4VLProcessor
    - __call__

## HYV4VLImageProcessor

[[autodoc]] HYV4VLImageProcessor

## HYV4VLImageProcessorPil

[[autodoc]] HYV4VLImageProcessorPil

## HYV4VLVideoProcessor

[[autodoc]] HYV4VLVideoProcessor
    - preprocess

## HYV4VLVisionModel

[[autodoc]] HYV4VLVisionModel
    - forward

## HYV4VLTextModel

[[autodoc]] HYV4VLTextModel
    - forward

## HYV4VLModel

[[autodoc]] HYV4VLModel
    - forward
    - get_image_embeddings
    - get_image_features
    - get_video_embeddings
    - get_video_features

## HYV4VLForConditionalGeneration

[[autodoc]] HYV4VLForConditionalGeneration
    - forward