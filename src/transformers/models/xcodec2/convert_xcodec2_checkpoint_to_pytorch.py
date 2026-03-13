# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Convert Xcodec2 checkpoints."""

import argparse
import json
import re

import safetensors
import torch

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    Xcodec2Config,
    Xcodec2FeatureExtractor,
    Xcodec2Model,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.xcodec2")

# Original key names: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py#L12
# Mappings are applied sequentially per key; order matters.
# fmt: off
STATE_DICT_MAPPING = {
    # ── Acoustic encoder (CodecEnc -> acoustic_encoder) ──
    r"^CodecEnc\.conv_blocks\.0\.":                                                 r"acoustic_encoder.initial_conv.",
    r"^CodecEnc\.conv_final_block\.0\.":                                            r"acoustic_encoder.final_activation.",
    r"^CodecEnc\.conv_final_block\.1\.":                                            r"acoustic_encoder.final_conv.",
    r"^CodecEnc\.conv_blocks\.(\d+)\.":                                             r"acoustic_encoder.encoder_blocks.SHIFT_DOWN.\1.",
    r"^CodecEnc\.":                                                                 r"acoustic_encoder.",
    # -- ResidualUnit: block.X.block.{0,1,2,3} -> block.X.{activation1,conv1,activation2,conv2}
    r"(\.block\.\d+)\.block\.0\.":                                                  r"\1.activation1.",
    r"(\.block\.\d+)\.block\.1\.":                                                  r"\1.conv1.",
    r"(\.block\.\d+)\.block\.2\.":                                                  r"\1.activation2.",
    r"(\.block\.\d+)\.block\.3\.":                                                  r"\1.conv2.",
    # -- EncoderBlock: block.{0,1,2} -> residual_units.{0,1,2}, block.3 -> activation, block.4 -> conv
    r"(encoder_blocks\.\d+)\.block\.([012])\.":                                     r"\1.residual_units.\2.",
    r"(encoder_blocks\.\d+)\.block\.3\.":                                           r"\1.activation.",
    r"(encoder_blocks\.\d+)\.block\.4\.":                                           r"\1.conv.",
    # -- DownSample1d: lowpass.filter -> filter (LowPassFilter1d inlined)
    r"\.downsample\.lowpass\.filter":                                               r".downsample.filter",

    # ── Quantizer: lives on main model, not inside decoder ──
    r"^generator\.quantizer\.layers\.0\.":                                          r"quantizer.fsq.",
    r"^generator\.quantizer\.":                                                     r"quantizer.",

    # ── Decoder (generator -> decoder) ──
    # -- Handle backbone components: now directly in decoder (no backbone. prefix)
    r"^generator\.backbone\.prior_net\.":                                           r"decoder.prior_blocks.",
    r"^generator\.backbone\.post_net\.":                                            r"decoder.post_blocks.",
    r"^generator\.backbone\.":                                                      r"decoder.",
    # -- General generator mapping
    r"^generator\.":                                                                r"decoder.",
    # -- Transformer layers
    r"\.att\.c_proj\.":                                                             r".self_attn.o_proj.",
    r"\.att_norm\.":                                                                r".input_layernorm.",
    r"\.ffn_norm\.":                                                                r".post_attention_layernorm.",

    # ── Semantic adapter (SemanticEncoder_module -> semantic_adapter) ──
    r"^SemanticEncoder_module\.":                                                   r"semantic_adapter.",
    r"semantic_adapter\.residual_blocks\.0\.":                                      r"semantic_adapter.act1.",
    r"semantic_adapter\.residual_blocks\.1\.":                                      r"semantic_adapter.conv1.",
    r"semantic_adapter\.residual_blocks\.2\.":                                      r"semantic_adapter.act2.",
    r"semantic_adapter\.residual_blocks\.3\.":                                      r"semantic_adapter.conv2.",
}
# fmt: on


def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    """
    When you go from the complex ROPE formulation to sin and cos one, you need
    to permute the query and key weights (to avoid doing it on the fly)
    """
    input_tensor = input_tensor.reshape(dim1, dim2)
    input_tensor = input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    input_tensor = input_tensor.transpose(1, 2).reshape(dim1, dim2)
    return input_tensor


def map_old_key_to_new(old_key):
    new_key = old_key
    for pattern, replacement in STATE_DICT_MAPPING.items():
        match = re.search(pattern, new_key)
        if match:
            # Handle index shift: conv_blocks index N (N>=1) -> encoder_blocks index N-1
            if "SHIFT_DOWN" in replacement and match.groups():
                layer_idx = int(match.group(1))
                replacement = replacement.replace(r"SHIFT_DOWN.\1", str(layer_idx - 1))
            new_key = re.sub(pattern, replacement, new_key)
    return new_key


def convert_state_dict(state_dict, n_heads):
    new_state_dict = {}

    for old_key, tensor in state_dict.items():
        # Special case: c_attn is a fused QKV projection that must be split into 3 tensors
        if ".att.c_attn." in old_key:
            w_q, w_k, w_v = torch.chunk(tensor, chunks=3, dim=0)
            w_q = permute_for_rope(w_q, n_heads, w_q.shape[0], w_q.shape[1])
            w_k = permute_for_rope(w_k, n_heads, w_k.shape[0], w_k.shape[1])
            base_key = map_old_key_to_new(old_key.replace(".att.c_attn.", ".self_attn.PROJ."))
            new_state_dict[base_key.replace("PROJ", "q_proj")] = w_q
            new_state_dict[base_key.replace("PROJ", "k_proj")] = w_k
            new_state_dict[base_key.replace("PROJ", "v_proj")] = w_v
            continue

        new_key = map_old_key_to_new(old_key)
        new_state_dict[new_key] = tensor
        if old_key != new_key:
            logger.debug(f"Converted: {old_key} -> {new_key}")

    return new_state_dict


def _convert_model(state_dict, hf_model):
    n_heads = hf_model.config.num_attention_heads
    state_dict = convert_state_dict(state_dict, n_heads)

    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    if len(extra_keys) != 0:
        raise ValueError(f"{len(extra_keys)} extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"{len(missing_keys)} missing keys found: {missing_keys}")
    hf_model.load_state_dict(state_dict, strict=True)

    n_params = sum(p.numel() for p in hf_model.parameters())
    logger.info(f"model loaded: {round(n_params / 1e6, 1)}M params")

    del state_dict
    return hf_model


@torch.no_grad()
def convert_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path=None,
    config_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    semantic_model_id = "facebook/w2v-bert-2.0"

    # load json
    if config_path is not None:
        with open(config_path, "r") as f:
            model_config = json.load(f)
        # # NOTE: `semantic_hidden_size` not needed as inside `semantic_model_config.hidden_size` (below)
        # semantic_hidden_size = model_config["semantic_hidden_size"]
        encoder_hidden_size = model_config["codec_encoder_hidden_size"]
        decoder_hidden_size = model_config["codec_decoder_hidden_size"]
        # # NOTE: not needed as always used: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py#L35
        # use_vocos = model_config["use_vocos"]
    else:
        # default to https://huggingface.co/HKUSTAudio/xcodec2/blob/main/config.json
        encoder_hidden_size = 1024
        decoder_hidden_size = 1024

    # create config
    # -- use hardcoded semantic model: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py#L19
    semantic_model_config = AutoConfig.from_pretrained(semantic_model_id, output_hidden_states=True)

    config = Xcodec2Config(
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        semantic_model_config=semantic_model_config,
    )

    # create model
    if not torch.cuda.is_available():
        raise ValueError("Run this script on a machine with a GPU for weight norm layers to be correctly copied.")
    torch_device = "cuda"
    model = Xcodec2Model(config).to(torch_device).eval()

    original_checkpoint = safetensors.torch.load_file(checkpoint_path)
    if "best_state" in original_checkpoint:
        # we might have a training state saved, in which case discard the yaml results and just retain the weights
        original_checkpoint = original_checkpoint["best_state"]

    # add weight norm, convert, and remove weight norm
    model.apply_weight_norm()
    model = _convert_model(original_checkpoint, model)
    model.remove_weight_norm()

    # create feature extractor
    dac_id = "descript/dac_16khz"
    semantic_model_id = "facebook/w2v-bert-2.0"
    dac_fe = AutoFeatureExtractor.from_pretrained(dac_id)
    semantic_fe = AutoFeatureExtractor.from_pretrained(semantic_model_id)
    if semantic_fe.sampling_rate != dac_fe.sampling_rate:
        raise ValueError(
            f"Sampling rates for DAC and semantic feature extractor must match, got {dac_fe.sampling_rate} and {semantic_fe.sampling_rate}."
        )
    feature_extractor = Xcodec2FeatureExtractor(
        feature_size=semantic_fe.feature_size,
        sampling_rate=semantic_fe.sampling_rate,
        num_mel_bins=semantic_fe.num_mel_bins,
        padding_value=semantic_fe.padding_value,
        stride=semantic_fe.stride,
        n_channels=dac_fe.feature_size,
        hop_length=dac_fe.hop_length,
        pre_padding_value=dac_fe.padding_value,
    )

    # save and upload
    if pytorch_dump_folder_path is not None:
        model.save_pretrained(pytorch_dump_folder_path)
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        feature_extractor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)


"""
While there is training code on GitHub: https://github.com/zhenye234/X-Codec-2.0
Modeling code on the Hub is used by xcodec2 (pip) package: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py

Given their example usage: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py
It seems best to use the config and weights from the Hub.

Conversion example usage:
```
# download model and config
wget https://huggingface.co/HKUSTAudio/xcodec2/resolve/main/config.json -P /raid/eric/xcodec2_original
wget https://huggingface.co/HKUSTAudio/xcodec2/resolve/main/model.safetensors -P /raid/eric/xcodec2_original

# run conversion
python src/transformers/models/xcodec2/convert_xcodec2_checkpoint_to_pytorch.py \
    --checkpoint_path /raid/eric/xcodec2_original/model.safetensors \
    --config_path /raid/eric/xcodec2_original/config.json \
    --push_to_hub hf-audio/xcodec2
```

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
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
