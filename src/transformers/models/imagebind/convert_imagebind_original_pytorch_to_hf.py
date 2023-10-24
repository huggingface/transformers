# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse

import torch
# from imagebind import load

from transformers import (
    ImageBindAudioConfig,
    ImageBindConfig,
    ImageBindDepthConfig,
    ImageBindImuConfig,
    ImageBindModel,
    ImageBindTextConfig,
    ImageBindThermalConfig,
    ImageBindVisionConfig,
)

SPATIOTEMPORAL_MODALITY_LIST = ["vision"]
IMAGELIKE_MODALITY_LIST = ["vision", "audio", "depth", "thermal"]
MODALITY_LIST = ["text", *IMAGELIKE_MODALITY_LIST, "imu"]


# Holds configs common to all test ImageBind encoders
IMAGEBIND_TEST_TRUNK_CONFIG = {
    "hidden_size": 32,
    "projection_dim": 32,
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "intermediate_size": 37,
    "dropout": 0.0,
    "layer_norm_eps": 1e-6,
}

IMAGEBIND_TEST_TEXT_CONFIG = {
    **IMAGEBIND_TEST_TRUNK_CONFIG,
    "vocab_size": 99,
    "logit_scale_init_value": 14.2857,
    "learnable_logit_scale": True,
}

IMAGEBIND_TEST_VISION_CONFIG = {
    **IMAGEBIND_TEST_TRUNK_CONFIG,
    "patch_size": (2, 2, 2),
    "stride": (2, 2, 2),
    "num_channels": 3,
    "num_frames": 2,
    "logit_scale_init_value": None,
    "learnable_logit_scale": False,
}

IMAGEBIND_TEST_AUDIO_CONFIG = {
    **IMAGEBIND_TEST_TRUNK_CONFIG,
    "patch_size": 4,
    "stride": 2,
    "num_channels": 1,
    "num_mel_bins": 128,
    "target_len": 204,
    "add_kv_bias": True,
    "drop_path_rate": 0.1,
    "logit_scale_init_value": 20.0,
    "learnable_logit_scale": False,
}

IMAGEBIND_TEST_DEPTH_CONFIG = {
    **IMAGEBIND_TEST_TRUNK_CONFIG,
    "patch_size": 2,
    "stride": 2,
    "num_channels": 1,
    "add_kv_bias": True,
    "logit_scale_init_value": 5.0,
    "learnable_logit_scale": False,
}

IMAGEBIND_TEST_THERMAL_CONFIG = {
    **IMAGEBIND_TEST_TRUNK_CONFIG,
    "patch_size": 2,
    "stride": 2,
    "num_channels": 1,
    "add_kv_bias": True,
    "logit_scale_init_value": 10.0,
    "learnable_logit_scale": False,
}

IMAGEBIND_TEST_IMU_CONFIG = {
    **IMAGEBIND_TEST_TRUNK_CONFIG,
    "input_shape": (6, 30),
    "kernel_size": 2,
    "add_kv_bias": True,
    "drop_path_rate": 0.7,
    "logit_scale_init_value": 5.0,
    "learnable_logit_scale": False,
}


def get_modality_config(config, modality):
    if modality == "text":
        return config.text_config
    elif modality == "vision":
        return config.vision_config
    elif modality == "audio":
        return config.audio_config
    elif modality == "depth":
        return config.depth_config
    elif modality == "thermal":
        return config.thermal_config
    elif modality == "imu":
        return config.imu_config
    else:
        raise ValueError(f"Modality {modality} is not currently supported.")


def convert_embeddings(config, model_state_dict):
    # Create position_ids buffer for text model]
    text_position_ids_buffer = torch.arange(config.text_config.max_position_embeddings).expand((1, -1))
    model_state_dict[f"text_model.embeddings.position_ids"] = text_position_ids_buffer

    # Create position_ids buffer for IMU model
    imu_num_patches = config.imu_config.input_shape[1] // config.imu_config.kernel_size
    imu_num_positions = imu_num_patches + 1
    imu_position_ids_buffer = torch.arange(imu_num_positions).expand((1, -1))
    model_state_dict[f"imu_model.embeddings.position_ids"] = imu_position_ids_buffer

    for modality in ["text", "imu"]:
        # Convert position embeddings for text and IMU modalities
        pos_embed_key = f"modality_preprocessors.{modality}.pos_embed"
        pos_embed = model_state_dict[pos_embed_key]
        converted_pos_embed = pos_embed.squeeze()
        model_state_dict[pos_embed_key] = converted_pos_embed

    for modality in IMAGELIKE_MODALITY_LIST:
        # Convert position embeddings for image-like modalities
        pos_embed_key = f"modality_preprocessors.{modality}.pos_embedding_helper.pos_embed"
        pos_embed = model_state_dict[pos_embed_key]
        converted_pos_embed = pos_embed.squeeze()
        model_state_dict[pos_embed_key] = converted_pos_embed

        # Create position_ids buffer for image-likd modalities
        modality_config = get_modality_config(config, modality)
        # Recalculate num_positions
        if modality in SPATIOTEMPORAL_MODALITY_LIST:
            patches_along_time_dim = modality_config.num_frames // modality_config.patch_size[0]
            patches_along_spatial_dims = (modality_config.image_size // modality_config.patch_size[1]) ** 2
            num_patches = patches_along_spatial_dims * patches_along_time_dim
        else:
            num_patches = (modality_config.image_size // modality_config.patch_size) ** 2
        num_positions = num_patches + 1
        position_ids_buffer = torch.arange(num_positions).expand((1, -1))
        model_state_dict[f"{modality}_model.embeddings.position_ids"] = position_ids_buffer

    for modality in IMAGELIKE_MODALITY_LIST + ["imu"]:
        # Convert class embeddings
        class_embed_key = f"modality_preprocessors.{modality}.cls_token"
        class_embed = model_state_dict[class_embed_key]
        converted_class_embed = class_embed.squeeze()
        model_state_dict[class_embed_key] = converted_class_embed


def convert_attention(config, model_state_dict):
    for modality in MODALITY_LIST:
        old_prefix = f"modality_trunks.{modality}.blocks"
        new_prefix = f"{modality}_model.encoder.layers"
        modality_config = get_modality_config(config, modality)
        for i in range(modality_config.num_hidden_layers):
            attn_weight_key = f"{old_prefix}.{i}.attn.in_proj_weight"
            attn_bias_key = f"{old_prefix}.{i}.attn.in_proj_bias"
            attn_weight = model_state_dict[attn_weight_key]
            attn_bias = model_state_dict[attn_bias_key]

            # Split up the attention projections/bias in to q, k, v projections/bias
            q_proj, k_proj, v_proj = attn_weight.chunk(3, dim=0)
            q_proj_bias, k_proj_bias, v_proj_bias = attn_bias.chunk(3, dim=0)

            model_state_dict[f"{new_prefix}.{i}.self_attn.q_proj.weight"] = q_proj
            model_state_dict[f"{new_prefix}.{i}.self_attn.q_proj.bias"] = q_proj_bias

            model_state_dict[f"{new_prefix}.{i}.self_attn.k_proj.weight"] = k_proj
            model_state_dict[f"{new_prefix}.{i}.self_attn.k_proj.bias"] = k_proj_bias

            model_state_dict[f"{new_prefix}.{i}.self_attn.v_proj.weight"] = v_proj
            model_state_dict[f"{new_prefix}.{i}.self_attn.v_proj.bias"] = v_proj_bias


def map_preprocessor_keys(prefix="modality_preprocessors"):
    mapping = {}
    keys_to_remove = []

    # Text preprocessor
    mapping[f"{prefix}.text.token_embedding.weight"] = "text_model.embeddings.token_embedding.weight"
    mapping[f"{prefix}.text.pos_embed"] = "text_model.embeddings.position_embedding.weight"

    # NOTE: no need to map causal attention mask buffer
    keys_to_remove.append("modality_preprocessors.text.mask")

    # Image-like modalities common
    for modality in IMAGELIKE_MODALITY_LIST:
        mapping[f"{prefix}.{modality}.cls_token"] = f"{modality}_model.embeddings.class_embedding"
        mapping[f"{prefix}.{modality}.pos_embedding_helper.pos_embed"] = f"{modality}_model.embeddings.position_embedding.weight"

    # Vision preprocessor specific
    mapping[f"{prefix}.vision.rgbt_stem.proj.1.weight"] = "vision_model.embeddings.patch_embedding.weight"

    # Audio preprocessor specific
    mapping[f"{prefix}.audio.rgbt_stem.proj.weight"] = "audio_model.embeddings.patch_embedding.weight"
    mapping[f"{prefix}.audio.rgbt_stem.norm_layer.weight"] = "audio_model.embeddings.norm_layer.weight"
    mapping[f"{prefix}.audio.rgbt_stem.norm_layer.bias"] = "audio_model.embeddings.norm_layer.bias"

    # Depth preprocessor specific
    mapping[f"{prefix}.depth.depth_stem.proj.weight"] = "depth_model.embeddings.patch_embedding.weight"
    mapping[f"{prefix}.depth.depth_stem.norm_layer.weight"] = "depth_model.embeddings.norm_layer.weight"
    mapping[f"{prefix}.depth.depth_stem.norm_layer.bias"] = "depth_model.embeddings.norm_layer.bias"

    # Thermal preprocessor specific
    mapping[f"{prefix}.thermal.rgbt_stem.proj.weight"] = "thermal_model.embeddings.patch_embedding.weight"
    mapping[f"{prefix}.thermal.rgbt_stem.norm_layer.weight"] = "thermal_model.embeddings.norm_layer.weight"
    mapping[f"{prefix}.thermal.rgbt_stem.norm_layer.bias"] = "thermal_model.embeddings.norm_layer.bias"

    # IMU preprocessor
    mapping[f"{prefix}.imu.cls_token"] = "imu_model.embeddings.class_embedding"
    mapping[f"{prefix}.imu.pos_embed"] = "imu_model.embeddings.position_embedding.weight"
    mapping[f"{prefix}.imu.imu_stem.proj.weight"] = "imu_model.embeddings.patch_embedding.weight"
    mapping[f"{prefix}.imu.imu_stem.norm_layer.weight"] = "imu_model.embeddings.norm_layer.weight"
    mapping[f"{prefix}.imu.imu_stem.norm_layer.bias"] = "imu_model.embeddings.norm_layer.bias"

    return mapping, keys_to_remove


def map_transformer_keys(config, old_prefix, new_prefix):
    mapping = {}
    keys_to_remove = []

    for i in range(config.num_hidden_layers):
        # NOTE: q, k, v proj/bias are added to the state dict with the correct names in convert_attention
        keys_to_remove.append(f"{old_prefix}.{i}.attn.in_proj_weight")
        keys_to_remove.append(f"{old_prefix}.{i}.attn.in_proj_bias")

        mapping[f"{old_prefix}.{i}.attn.out_proj.weight"] = f"{new_prefix}.{i}.self_attn.out_proj.weight"
        mapping[f"{old_prefix}.{i}.attn.out_proj.bias"] = f"{new_prefix}.{i}.self_attn.out_proj.bias"

        mapping[f"{old_prefix}.{i}.norm_1.weight"] = f"{new_prefix}.{i}.layer_norm1.weight"
        mapping[f"{old_prefix}.{i}.norm_1.bias"] = f"{new_prefix}.{i}.layer_norm1.bias"

        mapping[f"{old_prefix}.{i}.mlp.fc1.weight"] = f"{new_prefix}.{i}.mlp.fc1.weight"
        mapping[f"{old_prefix}.{i}.mlp.fc1.bias"] = f"{new_prefix}.{i}.mlp.fc1.bias"
        mapping[f"{old_prefix}.{i}.mlp.fc2.weight"] = f"{new_prefix}.{i}.mlp.fc2.weight"
        mapping[f"{old_prefix}.{i}.mlp.fc2.bias"] = f"{new_prefix}.{i}.mlp.fc2.bias"

        mapping[f"{old_prefix}.{i}.norm_2.weight"] = f"{new_prefix}.{i}.layer_norm2.weight"
        mapping[f"{old_prefix}.{i}.norm_2.bias"] = f"{new_prefix}.{i}.layer_norm2.bias"

        if config.add_kv_bias:
            mapping[f"{old_prefix}.{i}.attn.bias_k"] = f"{new_prefix}.{i}.self_attn.k_bias"
            mapping[f"{old_prefix}.{i}.attn.bias_v"] = f"{new_prefix}.{i}.self_attn.v_bias"

    return mapping, keys_to_remove


def get_encoder_key_mapping(config, prefix="modality_trunks"):
    mapping = {}
    keys_to_remove = []

    # 1. Handle any pre-transformer layers, if available.

    # Vision specific
    mapping["modality_trunks.vision.pre_transformer_layer.0.weight"] = "vision_model.pre_layernorm.weight"
    mapping["modality_trunks.vision.pre_transformer_layer.0.bias"] = "vision_model.pre_layernorm.bias"

    # 2. Map transformer trunk keys
    for modality in MODALITY_LIST:
        old_prefix = f"{prefix}.{modality}.blocks"
        new_prefix = f"{modality}_model.encoder.layers"
        modality_config = get_modality_config(config, modality)
        transformer_mapping, transformer_keys_to_remove = map_transformer_keys(modality_config, old_prefix, new_prefix)
        mapping.update(transformer_mapping)
        keys_to_remove.extend(transformer_keys_to_remove)

    return mapping, keys_to_remove


def map_transformer_head_keys(prefix="modality_heads"):
    mapping = {}
    keys_to_remove = []

    # Text final layer norm
    mapping[f"{prefix}.text.proj.0.weight"] = "text_model.final_layer_norm.weight"
    mapping[f"{prefix}.text.proj.0.bias"] = "text_model.final_layer_norm.bias"

    for modality in IMAGELIKE_MODALITY_LIST + ["imu"]:
        mapping[f"{prefix}.{modality}.0.weight"] = f"{modality}_model.post_layernorm.weight"
        mapping[f"{prefix}.{modality}.0.bias"] = f"{modality}_model.post_layernorm.bias"

    # Modality heads
    mapping[f"{prefix}.text.proj.1.weight"] = "text_projection.weight"
    for modality in IMAGELIKE_MODALITY_LIST:
        if modality == "vision":
            mapping[f"{prefix}.{modality}.2.weight"] = f"visual_projection.weight"
        else:
            mapping[f"{prefix}.{modality}.2.weight"] = f"{modality}_projection.weight"
    mapping[f"{prefix}.imu.3.weight"] = "imu_projection.weight"

    return mapping, keys_to_remove


def map_postprocessor_keys(prefix="modality_postprocessors"):
    mapping = {}
    keys_to_remove = []

    for modality in ["text", "audio", "depth", "thermal", "imu"]:
        mapping[f"{prefix}.{modality}.1.log_logit_scale"] = f"{modality}_postprocessor.log_logit_scale"

    return mapping, keys_to_remove


def get_key_mapping(config):
    mapping = {}
    keys_to_remove = []

    # 1. Map preprocessor keys
    preprocessor_mapping, preprocessor_keys_to_remove = map_preprocessor_keys(prefix="modality_preprocessors")
    mapping.update(preprocessor_mapping)
    keys_to_remove.extend(preprocessor_keys_to_remove)

    # 2. Map transformer keys
    encoder_mapping, encoder_keys_to_remove = get_encoder_key_mapping(config, prefix="modality_trunks")
    mapping.update(encoder_mapping)
    keys_to_remove.extend(encoder_keys_to_remove)

    # 3. Map transformer head keys
    head_mapping, head_keys_to_remove = map_transformer_head_keys(prefix="modality_heads")
    mapping.update(head_mapping)
    keys_to_remove.extend(head_keys_to_remove)

    # 4. Map postprocessor keys
    postprocessor_mapping, postprocessor_keys_to_remove = map_postprocessor_keys(prefix="modality_postprocessors")
    mapping.update(postprocessor_mapping)
    keys_to_remove.extend(postprocessor_keys_to_remove)

    return mapping, keys_to_remove


def rename_state_dict(state_dict, keys_to_modify, keys_to_remove):
    model_state_dict = {}
    for key, value in state_dict.items():
        if key in keys_to_remove:
            continue

        if key in keys_to_modify:
            new_key = keys_to_modify[key]
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value
    return model_state_dict


def convert_imagebind_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
    use_test_config=False,
    safe_serialization=False,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = ImageBindConfig.from_pretrained(config_path)
    elif use_test_config:
        config = ImageBindConfig(
            text_config=IMAGEBIND_TEST_TEXT_CONFIG,
            vision_config=IMAGEBIND_TEST_VISION_CONFIG,
            audio_config=IMAGEBIND_TEST_AUDIO_CONFIG,
            depth_config=IMAGEBIND_TEST_DEPTH_CONFIG,
            thermal_config=IMAGEBIND_TEST_THERMAL_CONFIG,
            imu_config=IMAGEBIND_TEST_IMU_CONFIG,
            projection_dim=32,
        )
    else:
        # The default config corresponds to the original ImageBind model.
        config = ImageBindConfig()

    hf_model = ImageBindModel(config)

    # print(hf_model)
    # hf_model_state_dict = hf_model.state_dict()
    # for key in hf_model_state_dict:
    #     print(key)

    # Original ImageBind checkpoint is a PyTorch state dict
    model_state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Fix embedding shapes
    convert_embeddings(config, model_state_dict)
    # Convert attention parameters to transformers
    convert_attention(config, model_state_dict)

    keys_to_modify, keys_to_remove = get_key_mapping(config)
    keys_to_remove = set(keys_to_remove)
    hf_state_dict = rename_state_dict(model_state_dict, keys_to_modify, keys_to_remove)

    hf_model.load_state_dict(hf_state_dict)

    hf_model.save_pretrained(pytorch_dump_folder_path, safe_serialization=safe_serialization)

    if repo_id:
        print("Pushing to the hub...")
        hf_model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to ImageBind checkpoint")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument("--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub.")
    parser.add_argument("--test", action="store_true", help="Whether to use the test config for ImageBind models.")
    parser.add_argument("--safe_serialization", action="store_true", help="Whether to save the model using `safetensors`.")

    args = parser.parse_args()

    convert_imagebind_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.push_to_hub,
        args.test,
        args.safe_serialization,
    )
