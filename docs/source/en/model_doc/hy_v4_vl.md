<!--Copyright 2026 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was contributed to Hugging Face Transformers on 2026-09-30.*

# HYV4VL

## Overview

HYV4VL combines the [HYV4](./hy_v4) mixture-of-experts text backbone with a vision encoder for
image-text and video-text generation. The text backbone uses Multi-head Latent Attention (MLA),
DeepSeek Sparse Attention (DSA), learned attention sinks, and independent Hyper-Connections (iHC).
It is not the dense, image-only [HunYuanVL](./hunyuan_vl) model.

The vision encoder uses learned positional embeddings interpolated to the input patch grid. A
patch merger applies RMS normalization, a strided convolution, GELU, and a second convolution
before inserting one newline embedding per merged row. Image and video features replace their
respective placeholder tokens in the language model input.

Multimodal rotary position embeddings interleave the time, height, and width axes. The
`mrope_section` entries must sum to half of `text_config.qk_rope_head_dim`; the default partition
is `[12, 10, 10]`. The query/key projection channels store adjacent real/imaginary pairs; the
model reorders these channels before applying split-layout rotary embeddings, both in MLA and
in the sparse-attention indexer. This channel layout is separate from the three-axis frequency
layout. Image and video grids include the additional newline column when constructing
language-model positions. Decode positions use the offset computed during multimodal prefill.

## Usage

The examples use a **local checkpoint directory**, not a published Hub model identifier. Set
`model_path` to a checkpoint with `model_type="hy_v4_vl"`, its tokenizer and chat template, and
its image and video processor configurations. The full model requires multiple GPUs; select a
tensor-parallel size that divides the checkpoint's sharded dimensions and leaves room for activations.

Run the following code on every rank of an initialized GPU allocation, with `RANK`, `LOCAL_RANK`,
`WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` set by your distributed launcher.

```python
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.distributed.configuration_utils import DistributedConfig

model_path = "/path/to/hy4-vl-checkpoint"
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=world_size),
).eval()

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "/path/to/image.png"},
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
    processor_kwargs={"images_kwargs": {"device": device}},
).to(device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
if int(os.environ["RANK"]) == 0:
    completion = outputs[0, inputs["input_ids"].shape[1]:]
    print(processor.decode(completion, skip_special_tokens=True))
```

For text-only generation, use a message containing only a `"text"` item. Keep the checkpoint's
chat template rather than constructing special tokens manually: token spellings and IDs are
checkpoint-specific.

The checkpoint template defaults to `reasoning_effort="high"`, which opens a reasoning segment.
Use its supported `reasoning_effort="no_think"` option for a direct answer, as above. When using
reasoning, allow enough generated tokens for both the reasoning segment and the answer;
a short token cap can end before the final answer begins. The vocabulary projection stays in
float32 when the rest of the model is loaded in bfloat16.

### Video inputs

The processor represents a clip as one contiguous visual-token span. It does not insert
per-frame timestamp text. Use it only with checkpoints trained for that input convention;
compatibility with timestamp-prefixed video checkpoints is not established.

Replace the message and preprocessing call in the example above with:

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "video", "video": "/path/to/video.mp4"},
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
        "videos_kwargs": {"do_sample_frames": True, "num_frames": 4, "fps": None, "device": device},
    },
).to(device)
```

Video-file decoding uses TorchCodec; install a TorchCodec version compatible with your PyTorch
build and its FFmpeg runtime dependencies. Sampling is disabled by default. Enable
`do_sample_frames` and pass either `num_frames` or `fps`, not both. When passing `num_frames`, set
`fps=None` to disable the processor's default frame rate. Frame-rate sampling requires
source frame-rate metadata. The processor limits sampled frames to `min_frames`, `max_frames`, and
the number of available frames.

### Image and video budgets

`HYV4VLImageProcessor` reuses the HunYuanVL image-processing implementation without overriding
its defaults. Load it from the checkpoint so the checkpoint's pixel budgets and normalization
settings take precedence over the inherited defaults.

The image processor configuration and `video_preprocessor_config.json` are independent:

- Image `size["longest_edge"]` is a per-image pixel budget.
- Video `size["longest_edge"]` is a pixel budget for the **whole clip**. Increasing the number of
  frames can reduce the spatial resolution of each frame.
- Video `max_image_size["longest_edge"]` additionally caps the pixel area of one frame; it is not
  an edge-length limit.
- `min_pixels` and `max_pixels` are aliases for the corresponding `size` budgets. Pass modality
  overrides through `images_kwargs` and `videos_kwargs` rather than changing both processors together.

A video grid `(t, h, w)` consumes `t * (h // merge_size) * (w // merge_size + 1)` visual tokens.
The extra column accounts for newline embeddings. Changing patch or merge sizes also changes the
model's visual layout; use the values that match the checkpoint's vision configuration.

### Supported execution paths

- The text backbone uses eager attention; the vision tower supports eager, SDPA, and Flash Attention
  dispatch. Select submodel backends independently rather than requesting SDPA for the entire model.
- Greedy, sampling, and beam expansion preserve each sample's packed image/video stream and the
  batch axis of three-dimensional position IDs.
- Assisted generation with returned iHC hidden states is limited by the shared generation output
  slicing utility. CPU/disk offload and long-context inference are not validated by the CUDA tests.
- The vision encoder requires `spatial_patch_size=1`. Patch and merge sizes must match the checkpoint.

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

## HYV4VLVisionTransformer

[[autodoc]] HYV4VLVisionTransformer
    - forward

## HYV4VLTextModel

[[autodoc]] HYV4VLTextModel
    - forward

## HYV4VLModel

[[autodoc]] HYV4VLModel
    - forward
    - get_image_features
    - get_video_features

## HYV4VLForConditionalGeneration

[[autodoc]] HYV4VLForConditionalGeneration
    - forward
