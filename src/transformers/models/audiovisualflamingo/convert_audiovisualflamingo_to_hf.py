# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Converts an original AudioVisualFlamingo checkpoint to the Transformers format.

The original checkpoint is a directory with a top-level `config.json` and one sub-directory per component:
`llm`, `vision_tower`, `mm_projector`, `sound_tower` and `sound_mm_projector`.

Usage:

```bash
python src/transformers/models/audiovisualflamingo/convert_audiovisualflamingo_to_hf.py \
    --input_dir /path/to/original/checkpoint --output_dir /path/to/hf/checkpoint
```
"""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from transformers import (
    AudioVisualFlamingoConfig,
    AudioVisualFlamingoForConditionalGeneration,
    AudioVisualFlamingoImageProcessor,
    AudioVisualFlamingoProcessor,
    AudioVisualFlamingoVideoProcessor,
    AutoTokenizer,
    GenerationConfig,
    Qwen2AudioEncoderConfig,
    Qwen2Config,
    SiglipVisionConfig,
    WhisperFeatureExtractor,
)


KEY_MAPPING = {
    r"^llm\.model\.": "model.language_model.",
    r"^llm\.lm_head\.": "lm_head.",
    r"^vision_tower\.vision_model\.": "model.vision_tower.",
    r"^sound_tower\.(audio_tower\.)?": "model.audio_tower.",
    r"^mm_projector\.layers\.1\.": "model.multi_modal_projector.layer_norm.",
    r"^mm_projector\.layers\.2\.": "model.multi_modal_projector.linear_1.",
    r"^mm_projector\.layers\.4\.": "model.multi_modal_projector.linear_2.",
    r"^sound_mm_projector\.layers\.0\.": "model.audio_projector.linear_1.",
    r"^sound_mm_projector\.layers\.2\.": "model.audio_projector.linear_2.",
}

# fmt: off
CHAT_TEMPLATE = (
    "{% macro render(content) %}"
        "{% if content is string %}{{ content }}"
        "{% else %}"
            "{% for item in content %}"
                "{% if item['type'] == 'text' %}{{ item['text'] }}"
                "{% elif item['type'] == 'image' %}{{ '<image>' }}"
                "{% elif item['type'] == 'video' %}{{ '<vila/video>' }}"
                "{% elif item['type'] == 'audio' %}{{ '<sound>' }}"
                "{% endif %}"
            "{% endfor %}"
        "{% endif %}"
    "{% endmacro %}"
    "{% if messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}"
    "{% endif %}"
    "{% for message in messages if message['content'] is not none %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' }}"
        "{% if message['role'] == 'assistant' %}"
            "{% generation %}{{ render(message['content']) + '<|im_end|>' }}{% endgeneration %}"
        "{% else %}"
            "{{ render(message['content']) + '<|im_end|>' }}"
        "{% endif %}"
        "{{ '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
# fmt: on


def load_sub_config(path: Path, config_class, **kwargs):
    """Loads a component config, keeping the fields of `config_class` but not the training-time settings or the
    `dtype`, which is set for the whole model."""
    config = json.loads((path / "config.json").read_text())
    ignored_keys = {"architectures", "dtype", "use_cache"}
    fields = config_class.__dataclass_fields__.keys() - ignored_keys
    return config_class(**{key: value for key, value in config.items() if key in fields} | kwargs)


def convert_config(input_dir: Path, original_config: dict, tokenizer) -> AudioVisualFlamingoConfig:
    video_encoder = original_config["video_encoder"]
    # The original LLM config keeps the special tokens of Qwen2.5, while the tokenizer has those AVF was trained with
    text_config = load_sub_config(
        input_dir / "llm",
        Qwen2Config,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return AudioVisualFlamingoConfig(
        text_config=text_config,
        vision_config=load_sub_config(input_dir / "vision_tower", SiglipVisionConfig),
        audio_config=load_sub_config(input_dir / "sound_tower", Qwen2AudioEncoderConfig),
        vision_feature_layer=original_config["mm_vision_select_layer"],
        s2_scales=[int(scale) for scale in str(original_config["s2_scales"]).split(",")],
        video_temporal_pool_size=video_encoder["pool_sizes"][0][0],
        time_embedding_dim=video_encoder["trope_dim"],
        # The original audio time embedding ignores `trope_theta` and keeps the default `audio_time_embedding_theta`
        video_time_embedding_theta=float(video_encoder["trope_theta"]),
        dtype="bfloat16",
    )


def convert_state_dict(input_dir: Path) -> dict[str, torch.Tensor]:
    state_dict = {}
    for component in ("llm", "vision_tower", "mm_projector", "sound_tower", "sound_mm_projector"):
        for file in sorted((input_dir / component).glob("*.safetensors")):
            for key, tensor in load_file(file).items():
                key = f"{component}.{key}"
                for pattern, replacement in KEY_MAPPING.items():
                    key, num_subs = re.subn(pattern, replacement, key)
                    if num_subs:
                        break
                else:
                    raise ValueError(f"Unexpected checkpoint key {key}")
                state_dict[key] = tensor
    return state_dict


def convert_processor(
    original_config: dict, config: AudioVisualFlamingoConfig, tokenizer
) -> AudioVisualFlamingoProcessor:
    tokenizer.model_max_length = config.text_config.max_position_embeddings
    return AudioVisualFlamingoProcessor(
        image_processor=AudioVisualFlamingoImageProcessor(
            s2_scales=config.s2_scales, max_tiles=original_config["max_tiles"]
        ),
        video_processor=AudioVisualFlamingoVideoProcessor(num_frames=original_config["num_video_frames"]),
        feature_extractor=WhisperFeatureExtractor(feature_size=config.audio_config.num_mel_bins),
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
        video_temporal_pool_size=config.video_temporal_pool_size,
        video_segment_duration=original_config["interleaved_video_segment_duration"],
        max_audio_len=int(original_config["audio_chunk_length"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory of the original checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the converted checkpoint to.")
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repository id to push the checkpoint to.")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    original_config = json.loads((input_dir / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(input_dir / "llm", padding_side="left")

    config = convert_config(input_dir, original_config, tokenizer)
    with torch.device("meta"):
        model = AudioVisualFlamingoForConditionalGeneration(config)
    model.load_state_dict(convert_state_dict(input_dir), strict=True, assign=True)
    model.generation_config = GenerationConfig.from_pretrained(input_dir / "llm")
    model.save_pretrained(args.output_dir)

    processor = convert_processor(original_config, config, tokenizer)
    processor.save_pretrained(args.output_dir)

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        processor.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
