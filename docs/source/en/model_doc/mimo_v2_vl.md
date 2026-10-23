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

Use a Transformers installation that provides [`MiMoV2VLForConditionalGeneration`], together with PyTorch,
torchvision, Accelerate, and Pillow. The audio/video examples also require a compatible TorchCodec installation
and FFmpeg. The examples download public media fixtures from the Hub and require network access on the first run.

The walkthrough below first preprocesses a message, then optionally loads the model to generate a response, and
finally saves and reloads both objects. Run the sections in order; the generation and saving sections reuse the
variables created in preprocessing.

### Preprocess images, audio, and video

This example loads only the processor and media, so it can run on a CPU without loading the model weights.
It uses the same public sample repositories as other Transformers multimodal examples and tests.

```python
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

model_id = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)

image_path = hf_hub_download(
    repo_id="huggingface/documentation-images",
    filename="pipeline-cat-chonk.jpeg",
    repo_type="dataset",
)
audio_path = hf_hub_download(
    repo_id="hf-internal-testing/transformers-synthetic-assets",
    filename="audio/voice_sample.wav",
    repo_type="dataset",
)
video_path = hf_hub_download(
    repo_id="hf-internal-testing/test-videos",
    filename="sample_demo_1_320x240.mp4",
    repo_type="dataset",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": image_path},
            {"type": "audio", "path": audio_path},
            {"type": "video", "path": video_path},
            {
                "type": "text",
                "text": "Describe the image, summarize the audio, and then describe the video.",
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
print({name: tuple(value.shape) for name, value in inputs.items()})
```

For a single modality, keep its content block and the text question. `hf_hub_download` returns real local files
and reuses the Hub cache on subsequent runs. For HTTP(S) resources, use `url` in place of `path`.
The processor also supports `image_url`, `audio_url`, and `video_url` content types: the corresponding field accepts
either a URL string or an object containing a `url` field. Normalizing these blocks does not modify the original messages.

The returned inputs include:

- `input_ids` and `attention_mask` for the expanded prompt.
- `pixel_values` and `image_grid_thw` for images.
- `pixel_values_videos` and `video_grid_thw` for video frames.
- `input_features`, `feature_attention_mask`, and `audio_feature_lens` for audio.
- `audio_segment_offsets` to align audio features with their prompt positions.

The processor loads waveform files at the configured sampling rate. When supplying waveform arrays directly,
resample them to that rate first. `audio_feature_lens` counts grouped audio tokens, not mel frames.

### Generate a response

Continue with `model_id`, `processor`, and `inputs` from the previous example:

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

generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
response = processor.batch_decode(
    generated_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
)[0]
print(response)
```

The original checkpoints contain TP-interleaved QKV and FP8/MXFP4 weights. The loading path converts them to
the requested floating-point dtype in memory and includes the learned audio tokenizer. Loading MiMo-V2.6-Flash-RL
in BF16 requires approximately 620 GB for weights, plus activation and cache memory, across the available devices.
This is a dequantized implementation; it does not use a serving framework's quantized inference kernels.
MTP/speculative-decoding weights are not required for this ordinary generation path.

Pass processor outputs directly to [`MiMoV2VLForConditionalGeneration.forward`] or `generate`.
The model converts mel features to codec IDs, encodes all supplied modalities, and replaces the corresponding
placeholder embeddings. During cached generation, media are encoded only during prefill.

### Save and reload

After running the generation example, save the model and processor to the same directory. The code releases the
original model before loading the saved copy to avoid keeping two full models in memory.

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
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.get_input_embeddings().weight.device)

generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])
```

Native saves use `model_type="mimo_v2_vl"` and include the audio tokenizer. Audio codebooks and router correction
biases retain FP32 precision when loading BF16/FP16 weights. RoPE initialization uses consistent arithmetic across
initial loading and reloading. Exact generated text can still depend on the device, kernels, and numerical settings.

To save only the processor, call `processor.save_pretrained()` without loading a model. Such a directory can be
reloaded with [`AutoProcessor`], but it does not contain model weights.

## Configuration and input notes

- Legacy text and multimodal checkpoints may both declare `model_type="mimo_v2"`. [`AutoConfig`] selects
  `mimo_v2_vl` when a nonempty vision/audio encoder configuration is present, and `mimo_v2_flash` otherwise.
- [`AutoProcessor`] recognizes legacy MiMo V2 VL checkpoints whose `preprocessor_config.json` still declares a
  Qwen processor. An explicit `processor_config.json` takes precedence.
- Native `layer_types` and `sliding_window` are supported. If `layer_types` and `hybrid_layer_pattern` are both
  supplied, they must agree. `rope_parameters` currently supports default RoPE; other scaling types are rejected.
- The original `generation_config.json` supplies EOS IDs and generation defaults. Inactive sampling settings
  that conflict with disabled sampling are cleared for native serialization. When enabling sampling, pass the
  desired options explicitly, such as `do_sample=True, temperature=1.0, top_p=0.95`.
- Pre-tokenized audio uses `is_tokenized=True` with `(time, codec_channels)` integer arrays. The native model accepts
  the resulting `input_audio_codes` directly. `prepare_model_inputs()` is an optional adapter for legacy input names.
- Decoded video arrays require `video_metadata` containing the source frame rate and frame indices. Use
  `do_sample_frames=False` for already-sampled frames.
- Video blocks process frames by default. To interleave a video's soundtrack, provide the decoded track through
  `audio` in a direct processor call and set `use_audio_in_video=True`. Audio inputs follow placeholder order.
  `video_audio_interleave_length` selects per-grid (`0`), video-then-audio (`-1`), or positive-duration grouping.
- `audio_feature_lens` accounts for the learned tokenizer's `audio_segment_size`. Padding frames are excluded before
  audio segmentation. For training templates with Jinja `generation` blocks, `return_assistant_tokens_mask=True`
  uses replacement offsets to keep assistant spans aligned after multimodal placeholder expansion.

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
