<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# MiMo V2 VL

## Overview

MiMo V2 VL generates text from text, images, audio, and video. It combines a mixture-of-experts language decoder
with a visual encoder, audio codec embeddings, and a learned audio tokenizer. The language decoder alternates global
and sliding-window attention with partial RoPE and attention sinks. The visual encoder processes spatial/temporal
patches with row- and column-ordered attention windows.

The following checkpoints share this implementation and provide their own model and preprocessing configurations:

- [XiaomiMiMo/MiMo-V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5)
- [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)
- [XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)

For the text-only MiMo-V2-Flash and MiMo-V2.5-Pro checkpoints, see [MiMo-V2-Flash](mimo_v2_flash).

## Usage

Install Transformers from source with PyTorch, torchvision, Pillow, and Accelerate. For audio and video files,
install [TorchCodec](https://github.com/pytorch/torchcodec#installing-torchcodec) 0.5 or newer and its FFmpeg dependencies,
using a TorchCodec version compatible with your PyTorch installation.

### Image inputs

Use [`AutoProcessor`] to load the checkpoint's tokenizer, chat template, and preprocessing configuration.
The following example runs on CPU and downloads only the processor files and a sample image.

```python
from transformers import AutoProcessor

model_id = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
print({name: tuple(value.shape) for name, value in inputs.items()})
```

For a local image, replace the image block with `{"type": "image", "path": "/path/to/image.jpg"}`.
These examples set `enable_thinking=False` to request a direct answer. To enable reasoning, remove this option
and increase `max_new_tokens` to leave room for both reasoning and the final answer.

### Generate a response

Load the model and pass the processor outputs directly to `generate`. This section uses `model_id`, `processor`,
and `inputs` from the image example. It also works with the audio, video, and mixed inputs shown below.

The original checkpoint loader dequantizes FP8/MXFP4 weights to the requested floating-point dtype. For example,
MiMo-V2.6-Flash-RL needs approximately 620 GB for BF16 weights, plus activation and cache memory. Use a multi-device
setup with sufficient memory. Native quantized inference and MTP/speculative decoding are not implemented by this loader.

```python
import torch
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=False,
)
inputs = inputs.to(model.get_input_embeddings().weight.device)

generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False, logits_to_keep=1)
response = processor.batch_decode(
    generated_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
)[0]
print(response)
```

The model encodes images, converts audio mel features to codec IDs with its learned audio tokenizer, and fuses
the resulting features with the text embeddings. During cached generation, media are encoded only during prefill.

### Audio and video inputs

The following examples preprocess audio or video independently. They use public fixtures also used by the
Transformers processor tests. Each tab can be run on CPU without loading model weights.

<hfoptions id="media">
<hfoption id="Audio">

```python
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

model_id = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
audio_path = hf_hub_download(
    repo_id="hf-internal-testing/dummy-audio-samples",
    filename="f2641_0_throatclearing.wav",
    repo_type="dataset",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "path": audio_path},
            {"type": "text", "text": "Describe the sound in this audio."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt"
)
print(inputs.input_features.shape, inputs.audio_feature_lens)
```

</hfoption>
<hfoption id="Video">

```python
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

model_id = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
video_path = hf_hub_download(
    repo_id="hf-internal-testing/test-videos",
    filename="tiny_video_320x240.mp4",
    repo_type="dataset",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": video_path},
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt"
)
print(inputs.pixel_values_videos.shape, inputs.video_grid_thw)
```

</hfoption>
</hfoptions>

`hf_hub_download` caches the media locally. Audio files are resampled to the checkpoint's sampling rate. When
passing waveform arrays directly to the processor, resample them first and pass the matching `sampling_rate`.
Video content blocks process frames; they do not automatically include the video's soundtrack.

### Mixed inputs

Combine image, audio, video, and text content blocks in the same message. After running the examples above,
the following uses the downloaded `audio_path` and `video_path`:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "audio", "path": audio_path},
            {"type": "video", "path": video_path},
            {"type": "text", "text": "Describe the image, the sound, and the video in that order."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt"
)
```

For batches, pass a list of conversations and set `processor_kwargs={"text_kwargs": {"padding": True}}`.
The processor also accepts `image_url`, `audio_url`, and `video_url` content types with a string or a nested
`{"url": ...}` value, for example `{"type": "image_url", "image_url": {"url": "https://..."}}`.
Applying the chat template leaves the original messages unchanged.

### Save and reload

Save the model and processor to the same directory after generation. The saved weights use the native
`mimo_v2_vl` format and include the audio tokenizer. A BF16 export requires approximately as much disk space
as the in-memory weights. Release the original model before loading the saved copy.

```python
import gc
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

output_dir = "./mimo-v2-vl-native"
model.save_pretrained(output_dir, max_shard_size="5GB")
processor.save_pretrained(output_dir)

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

processor = AutoProcessor.from_pretrained(output_dir, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    output_dir,
    device_map="auto",
    dtype="auto",
    local_files_only=True,
)
```

Preprocess `messages` again and use the generation example with the reloaded objects. To save only the processor,
call `processor.save_pretrained()`; the resulting directory can be loaded with [`AutoProcessor`] on CPU.

## Configuration and input notes

- Legacy text and multimodal checkpoints may both declare `model_type="mimo_v2"`. [`AutoConfig`] selects
  `mimo_v2_vl` when a nonempty vision/audio encoder configuration is present, and `mimo_v2_flash` otherwise.
- [`AutoProcessor`] recognizes legacy MiMo V2 VL checkpoints whose `preprocessor_config.json` still declares a
  Qwen processor. An explicit `processor_config.json` takes precedence.
- The returned model inputs include `pixel_values` / `image_grid_thw` for images, `pixel_values_videos` /
  `video_grid_thw` for videos, and `input_features` / `feature_attention_mask` for waveforms. `audio_feature_lens`
  counts grouped audio tokens, and `audio_segment_offsets` aligns them with the prompt.
- The original `generation_config.json` supplies EOS IDs and generation defaults. When enabling sampling, pass
  the desired options explicitly, such as `do_sample=True, temperature=1.0, top_p=0.95`.
- Pre-tokenized audio uses `is_tokenized=True` with `(time, codec_channels)` integer arrays. The native model accepts
  the resulting `input_audio_codes` directly. `prepare_model_inputs()` is an optional adapter for legacy input names.
- Decoded video arrays require `video_metadata` containing the source frame rate and frame indices. Use
  `do_sample_frames=False` for already-sampled frames.
- Video blocks process frames by default. To interleave a video's soundtrack, provide the decoded track through
  `audio` in a direct processor call and set `use_audio_in_video=True`. Audio inputs follow placeholder order.
  `video_audio_interleave_length` selects per-grid (`0`), video-then-audio (`-1`), or positive-duration grouping.

## MiMoV2VLConfig

[[autodoc]] MiMoV2VLConfig

## MiMoV2VLTextConfig

[[autodoc]] MiMoV2VLTextConfig

## MiMoV2VLVisionConfig

[[autodoc]] MiMoV2VLVisionConfig

## MiMoV2VLAudioConfig

[[autodoc]] MiMoV2VLAudioConfig

## MiMoV2VLAudioTokenizerConfig

[[autodoc]] MiMoV2VLAudioTokenizerConfig

## MiMoV2VLModel

[[autodoc]] MiMoV2VLModel
    - forward
    - get_image_features
    - get_video_features
    - get_audio_features

## MiMoV2VLTextModel

[[autodoc]] MiMoV2VLTextModel

## MiMoV2VLForConditionalGeneration

[[autodoc]] MiMoV2VLForConditionalGeneration
    - forward
    - from_original_checkpoint

## MiMoV2Processor

[[autodoc]] MiMoV2Processor
    - __call__
    - prepare_model_inputs

## MiMoV2ImageProcessor

[[autodoc]] MiMoV2ImageProcessor
    - preprocess

## MiMoV2VideoProcessor

[[autodoc]] MiMoV2VideoProcessor
    - preprocess

## MiMoV2FeatureExtractor

[[autodoc]] MiMoV2FeatureExtractor
    - __call__
