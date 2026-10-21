<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# MiMo V2 VL

MiMo V2 VL is the multimodal model family for text generation from text, images, videos, and audio.
The family includes the following checkpoints, which share the multimodal architecture:

- [XiaomiMiMo/MiMo-V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5)
- [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)
- [XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)

The same implementation reads each checkpoint's dimensions, layer patterns, normalization, router dtype, and
preprocessing settings. The examples below use MiMo-V2.6-Flash-RL; replace the checkpoint ID to use another family member.

| Checkpoint | Language hidden size | Language layers | Language norm epsilon | Router compute dtype |
| --- | --- | --- | --- | --- |
| MiMo-V2.5 | 4096 | 48 | `1e-5` | float32 |
| MiMo-V2.6-Flash-RL | 4096 | 48 | `1e-6` | bfloat16 |
| MiMo-V2.6-Pro-RL | 6144 | 70 | `1e-5` | bfloat16 |

The architecture includes:

- A MoE language decoder with global/sliding-window attention, partial RoPE, attention sinks, and different key/value dimensions.
- A visual encoder with grouped-query attention, alternating row/column sliding windows, two-dimensional RoPE, and a spatial patch merger.
- Per-channel audio codec embeddings, a local audio transformer, and a projection into the language hidden space.
- The learned mel-to-codec encoder and residual vector quantizer distributed with the checkpoint.

The model encodes media and replaces the corresponding placeholder embeddings before the language decoder runs.
During cached generation it encodes media only during prefill. Beam expansion preserves the association between
each prompt and its flattened visual/audio features.

## Automatic model selection

The original text and multimodal checkpoints can both declare `model_type="mimo_v2"` and
`architectures=["MiMoV2ForCausalLM"]`. [`AutoConfig`] resolves this legacy model type from the encoder settings:

| Configuration | Selected model family |
| --- | --- |
| No nonempty `vision_config` or `audio_config` | `mimo_v2_flash`, including MiMo-V2.5-Pro |
| A nonempty `vision_config` or `audio_config` | `mimo_v2_vl`, including MiMo-V2.5, MiMo-V2.6-Flash-RL, and MiMo-V2.6-Pro-RL |

Native `mimo_v2_flash` and `mimo_v2_vl` model types remain explicit choices. Repository names and the shared
`architectures` string are not used to distinguish these families. Sliding-attention dimensions, layer patterns,
normalization and RoPE settings are read from each checkpoint.

Use the same Auto entry point for either family:

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

checkpoint = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, device_map="auto", dtype=torch.bfloat16, trust_remote_code=False
)
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=False)
```

For legacy MiMo V2 VL repositories, [`AutoProcessor`] corrects the Qwen processor declaration in
`preprocessor_config.json`. An explicit `processor_config.json` takes precedence. Text-only checkpoints use their
tokenizer rather than requiring a multimodal processor.

## Loading original weights

Released checkpoints use TP-interleaved QKV weights with checkpoint-specific FP8 or packed MXFP4 storage.
The Auto loading path detects the original format and streams dequantized weights to the selected devices.
You can also use the explicit multimodal class:

```python
import torch
from transformers import MiMoV2Processor, MiMoV2VLForConditionalGeneration

checkpoint = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
processor = MiMoV2Processor.from_pretrained(checkpoint)
model = MiMoV2VLForConditionalGeneration.from_original_checkpoint(
    checkpoint,
    device_map="auto",
    dtype=torch.bfloat16,
)
```

The loader includes the audio tokenizer weights from the checkpoint's `audio_tokenizer` subdirectory and verifies
that all required model tensors were loaded. MTP/speculative-decoding weights are not used for ordinary generation.
Loading requires Accelerate and enough aggregate CPU/GPU memory for the dequantized model. It does not create a
second checkpoint on disk. This implementation executes the dequantized model, rather than SGLang's quantized kernels.

Native saved checkpoints use `model_type="mimo_v2_vl"` and also work with the image/text Auto class:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model.save_pretrained("./mimo-v2-vl", max_shard_size="5GB")
processor.save_pretrained("./mimo-v2-vl")

processor = AutoProcessor.from_pretrained("./mimo-v2-vl")
model = AutoModelForImageTextToText.from_pretrained("./mimo-v2-vl", device_map="auto", dtype="auto")
```

For offline conversion:

```bash
python -m transformers.models.mimo_v2_vl.convert_mimo_v2_vl_weights_to_hf \
  --checkpoint /path/to/MiMo-V2.6-Flash-RL \
  --output /path/to/native-mimo-v2-vl --device cpu
```

## Image understanding

```python
from transformers.image_utils import load_image

image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Describe this image."},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.get_input_embeddings().weight.device)
generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Audio and video

Pass mono audio at the configured sampling rate (24 kHz for the checkpoints listed above). The processor computes mel
features; the model runs the learned audio tokenizer, embeds its codec IDs, and inserts the projected audio features.

```python
import numpy as np

waveform = np.zeros(24000, dtype=np.float32)  # Replace with a decoded 24 kHz mono waveform.
prompt = processor.apply_chat_template(
    [{"role": "user", "content": processor.audio_token + " Describe the audio."}],
    tokenize=False, add_generation_prompt=True,
)
inputs = processor(text=prompt, audio=waveform, sampling_rate=24000, return_tensors="pt")
inputs = inputs.to(model.get_input_embeddings().weight.device)
generated_ids = model.generate(**inputs, max_new_tokens=64)
```

Pre-tokenized audio is supported with `is_tokenized=True`; each sequence has shape `(time, codec_channels)`.
The native model directly accepts `input_audio_codes`, `audio_feature_lens`, and `audio_segment_offsets` from the
processor. `prepare_model_inputs` is optional: its legacy `audio_codes` and `video_pixel_values` names are also accepted.

Decoded video frames require `video_metadata` with the original frame rate and frame indices. Use
`do_sample_frames=False` when frames are already sampled. To include a video's audio track, provide it in `audio`
and set `use_audio_in_video=True`, or pass one flag per video. Audio sequences follow placeholder order across the
text batch. `video_audio_interleave_length` selects per-grid (`0`), video-then-audio (`-1`), or positive-duration
time-window grouping. The model consumes the processor's offsets to insert the matching audio segments.

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
    - forward

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
