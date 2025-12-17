# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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
import gc
import re

import safetensors.torch

from transformers.models.pe_audio_video.modeling_pe_audio_video import PeAudioVideoConfig, PeAudioVideoModel
from transformers.utils import cached_file


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"audio_video_model\.audio_model\.dac_vae_encoder\.encoder": "audio_video_encoder.audio_encoder.dac_encoder",
    r"audio_video_model\.audio_model\.dac_vae_encoder\.bottleneck\.in_proj": "audio_video_encoder.audio_encoder.bottleneck",
    r"audio_video_model\.audio_model\.data_proj": "audio_video_encoder.audio_encoder.data_proj",
    r"audio_video_model\.audio_model\.transformer\.embeddings\.resnet_block": "audio_video_encoder.audio_encoder.embeddings.resnet_block",
    r"audio_video_model\.audio_model\.transformer\.embeddings\.cls_token": "audio_video_encoder.audio_encoder.embeddings.class_embedding",
    r"audio_video_model\.audio_model\.transformer\.layers": "audio_video_encoder.audio_encoder.layers",
    r"audio_video_model\.audio_model\.transformer\.norm": "audio_video_encoder.audio_encoder.norm",
    r"audio_video_model\.audio_model\.transformer\.output": "audio_video_encoder.audio_encoder.output",
    r"audio_video_model\.video_model\.clip_vision_model": "audio_video_encoder.video_encoder.vision_model",
    r"audio_video_model\.video_model\.proj": "audio_video_encoder.video_encoder.proj",
    r"audio_video_model\.video_model\.data_proj": "audio_video_encoder.video_encoder.data_proj",
    r"audio_video_model\.video_model\.transformer\.embeddings\.resnet_block": "audio_video_encoder.video_encoder.embeddings.resnet_block",
    r"audio_video_model\.video_model\.transformer\.embeddings\.cls_token": "audio_video_encoder.video_encoder.embeddings.class_embedding",
    r"audio_video_model\.video_model\.transformer\.layers": "audio_video_encoder.video_encoder.layers",
    r"audio_video_model\.video_model\.transformer\.norm": "audio_video_encoder.video_encoder.norm",
    r"audio_video_model\.video_model\.transformer\.output": "audio_video_encoder.video_encoder.output",
    r"audio_video_model\.transformer\.embeddings\.resnet_block": "audio_video_encoder.embeddings.resnet_block",
    r"audio_video_model\.transformer\.embeddings\.cls_token": "audio_video_encoder.embeddings.class_embedding",
    r"audio_video_model\.transformer\.layers": "audio_video_encoder.layers",
    r"audio_video_model\.transformer\.norm": "audio_video_encoder.norm",
    r"audio_video_model\.transformer\.output": "audio_video_encoder.output",
    r"audio_video_model\.transformer\.modality_aligner.conv": "audio_video_encoder.video_proj",
    r"audio_video_model\.transformer\.modality_aligner.layer_norm": "audio_video_encoder.video_norm",
    r"audio_video_model\.transformer\.concat_modality_proj": "audio_video_encoder.concat_modality_proj",
    r"audio_video_model\.transformer\.data_proj": "audio_video_encoder.data_proj",
    r"audio_video_text_head": "text_head_audio_video",
    r"audio_text_head": "text_head_audio",
    r"video_text_head": "text_head_video",
}

path = cached_file("facebook/pe-av-large", "model.safetensors")
state_dict = safetensors.torch.load_file(path)

config = PeAudioVideoConfig()
model = PeAudioVideoModel(config)


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def permute(w, n_heads, dim1, dim2):
    """
    Permute weights for rotary embeddings.
    Based on convert_perception_lm_weights_to_hf.py line 227-228
    """
    # return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    return w


converted_state_dict = {}
for original_key, tensor in state_dict.items():
    if "out_proj" in original_key:
        # this is not used and should be ignored
        continue

    if "text_model" in original_key:
        converted_state_dict[original_key] = tensor
        continue
    elif "audio_model" in original_key:
        current_config = config.audio_video_config.audio_config
    elif "video_model" in original_key:
        current_config = config.audio_video_config.video_config
    else:
        current_config = None

    if current_config is not None:
        # Get config parameters for permutation
        n_heads = current_config.num_attention_heads
        num_key_value_heads = current_config.num_key_value_heads
        hidden_size = current_config.hidden_size
        head_dim = getattr(current_config, "head_dim", hidden_size // n_heads)

        # Calculate dimensions
        dim = n_heads * head_dim
        key_value_dim = num_key_value_heads * head_dim

    converted_key = convert_key(original_key, ORIGINAL_TO_CONVERTED_KEY_MAPPING)
    if ".self_attn.q_proj.weight" in converted_key:
        converted_state_dict[converted_key] = permute(tensor, n_heads=n_heads, dim1=dim, dim2=hidden_size)
    elif ".self_attn.k_proj.weight" in converted_key:
        converted_state_dict[converted_key] = permute(
            tensor, n_heads=num_key_value_heads, dim1=key_value_dim, dim2=hidden_size
        )
    else:
        converted_state_dict[converted_key] = tensor

model.load_state_dict(converted_state_dict, strict=True, assign=True)
del model.config._name_or_path

print("Saving the model.")
model.save_pretrained("/raid/eustache/sam-audio/converted", safe_serialization=True)
del state_dict, model

# Safety check: reload the converted model
gc.collect()
print("Reloading the model to check if it's saved correctly.")
PeAudioVideoModel.from_pretrained("/raid/eustache/sam-audio/converted", device_map="auto")
print("Model reloaded successfully.")
