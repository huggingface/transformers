# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Convert Mimi checkpoints."""

import argparse

import safetensors
import torch

from transformers import (
    EncodecFeatureExtractor,
    MimiConfig,
    MimiModel,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.mimi")


def assert_param_count(model_1, model_2):
    count_1 = sum(p[1].numel() for p in model_1.named_parameters() if "final_proj" not in p[0])
    count_2 = sum(p[1].numel() for p in model_2.named_parameters() if "final_proj" not in p[0])
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"


def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])


def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


convert_list = [
    # GENERAL
    ("conv.conv.conv", "conv"),
    ("convtr.convtr.convtr", "conv"),
    ("conv.conv", "conv"),
    ("convtr.convtr", "conv"),
    # QUANTIZER
    ("quantizer.rvq_first.vq", "quantizer.semantic_residual_vector_quantizer"),
    ("quantizer.rvq_first", "quantizer.semantic_residual_vector_quantizer"),
    ("quantizer.rvq_rest.vq", "quantizer.acoustic_residual_vector_quantizer"),
    ("quantizer.rvq_rest", "quantizer.acoustic_residual_vector_quantizer"),
    ("_codebook", "codebook"),
    ("_initialized", "initialized"),
    ("embedding_sum", "embed_sum"),
    # ENCODER PART
    ("encoder.model", "encoder.layers"),
    ("decoder.model", "decoder.layers"),
    # TRANSFORMERS PART
    ("encoder_transformer.transformer", "encoder_transformer"),
    ("decoder_transformer.transformer", "decoder_transformer"),
    ("linear1", "mlp.fc1"),
    ("linear2", "mlp.fc2"),
    ("self_attn.out_proj", "self_attn.o_proj"),
    ("norm1", "input_layernorm"),
    ("norm2", "post_attention_layernorm"),
    ("layer_scale_1", "self_attn_layer_scale"),
    ("layer_scale_2", "mlp_layer_scale"),
]


def _convert_model(
    state_dict,
    hf_model,
    convert_list,
    device,
    config,
    unwanted_prefix=None,
):
    hidden_size = config.hidden_size
    head_dim = config.head_dim
    num_heads = int(config.hidden_size // config.head_dim)
    num_key_value_heads = config.num_key_value_heads
    key_value_head_dim = config.num_key_value_heads * head_dim

    # permute for sliced rotary
    def permute(w, n_heads, dim1=hidden_size, dim2=hidden_size):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    for k, v in list(state_dict.items()):
        new_k = k if unwanted_prefix is None else k[len(unwanted_prefix) :]
        for old_layer_name, new_layer_name in convert_list:
            if old_layer_name in new_k:
                new_k = new_k.replace(old_layer_name, new_layer_name)

        if "in_proj_weight" in new_k:
            # split qkv into query key and value
            mixed_qkv = state_dict.pop(k)
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            state_dict[new_k.replace("in_proj_weight", "q_proj.weight")] = permute(query_layer, num_heads)
            state_dict[new_k.replace("in_proj_weight", "k_proj.weight")] = permute(
                key_layer, num_key_value_heads, dim1=key_value_head_dim
            )
            state_dict[new_k.replace("in_proj_weight", "v_proj.weight")] = value_layer
        else:
            state_dict[new_k] = state_dict.pop(k)

    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    hf_model.load_state_dict(state_dict, strict=True)
    n_params = param_count(hf_model)

    logger.info(f"model loaded: {round(n_params / 1e6, 1)}M params")

    hf_model.eval()
    hf_model.to(device)
    del state_dict

    return hf_model


@torch.no_grad()
def convert_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    device = _grab_best_device()

    if config_path is not None:
        config = MimiConfig.from_pretrained(config_path)
    else:
        config = MimiConfig()

    model = MimiModel(config)

    feature_extractor = EncodecFeatureExtractor(
        feature_size=config.audio_channels,
        sampling_rate=config.sampling_rate,
    )
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    original_checkpoint = safetensors.torch.load_file(checkpoint_path)
    if "best_state" in original_checkpoint:
        # we might have a training state saved, in which case discard the yaml results and just retain the weights
        original_checkpoint = original_checkpoint["best_state"]

    model = _convert_model(original_checkpoint, model, convert_list, device, config)

    model.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        feature_extractor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.push_to_hub,
    )
