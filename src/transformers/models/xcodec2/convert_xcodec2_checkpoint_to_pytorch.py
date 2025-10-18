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


def assert_param_count(model_1, model_2):
    count_1 = sum(p[1].numel() for p in model_1.named_parameters() if "final_proj" not in p[0])
    count_2 = sum(p[1].numel() for p in model_2.named_parameters() if "final_proj" not in p[0])
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"


def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])


def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    """
    When you go from the complex ROPE formulation to sin and cos one, you need
    to permute the query and key weights (to avoid doing it on the fly)
    """
    input_tensor = input_tensor.reshape(dim1, dim2)
    input_tensor = input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    input_tensor = input_tensor.transpose(1, 2).reshape(dim1, dim2)
    return input_tensor


def _convert_model(state_dict, hf_model):
    tensors = {}

    for old_k in state_dict.keys():
        # update to new names used in `Xcodec2Model` in modeling_xcodec2.py
        # old names can be found here: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py#L12
        if "CodecEnc" in old_k:
            k = old_k.replace("CodecEnc", "acoustic_encoder")

            # Handle initial convolutional layer (conv_blocks.0 -> initial_conv)
            if "conv_blocks.0" in k:
                k = k.replace("conv_blocks.0", "initial_conv")

            # Handle final layers (conv_final_block.0 -> final_activation, conv_final_block.1 -> final_conv)
            elif "conv_final_block.0" in k:
                k = k.replace("conv_final_block.0", "final_activation")
            elif "conv_final_block.1" in k:
                k = k.replace("conv_final_block.1", "final_conv")

            # Handle encoder blocks (conv_blocks.1, conv_blocks.2, etc. -> encoder_blocks.0, encoder_blocks.1, etc.)
            elif "conv_blocks." in k:
                # Extract the block index (subtracting 1 because initial_conv replaced conv_blocks.0)
                conv_block_pattern = r"acoustic_encoder\.conv_blocks\.(\d+)(.*)"
                match = re.match(conv_block_pattern, k)
                if match:
                    block_idx = int(match.group(1))
                    suffix = match.group(2)
                    # Adjust index (subtract 1 because we moved conv_blocks.0 to initial_conv)
                    k = f"acoustic_encoder.encoder_blocks.{block_idx - 1}{suffix}"

            # Handle the ResidualUnit structure changes (nested blocks)
            # We're looking for patterns like "block.X.block.Y" where Y is 0-3
            nested_block_pattern = r"(.*\.block\.\d+)\.block\.(\d+)(.*)"
            match = re.match(nested_block_pattern, k)
            if match:
                prefix = match.group(1)  # Everything before .block.Y
                subblock_idx = int(match.group(2))  # Y value (0-3)
                suffix = match.group(3)  # Everything after .block.Y

                # Map to the corresponding component in ResidualUnit
                if subblock_idx == 0:
                    k = f"{prefix}.activation1{suffix}"
                elif subblock_idx == 1:
                    k = f"{prefix}.conv1{suffix}"
                elif subblock_idx == 2:
                    k = f"{prefix}.activation2{suffix}"
                elif subblock_idx == 3:
                    k = f"{prefix}.conv2{suffix}"

            # Handle the EncoderBlock structure changes
            # We're looking for patterns like "encoder_blocks.X.block.Y" where Y is 0-2 (residual units), 3 (activation), or 4 (conv)
            encoder_block_pattern = r"(acoustic_encoder\.encoder_blocks\.\d+)\.block\.(\d+)(.*)"
            match = re.match(encoder_block_pattern, k)
            if match:
                prefix = match.group(1)  # Everything before .block.Y (conv_blocks.X)
                block_idx = int(match.group(2))  # Y value (0-4)
                suffix = match.group(3)  # Everything after .block.Y

                # Map to the corresponding component in EncoderBlock
                if block_idx < 3:  # First 3 are residual units (typically 0, 1, 2)
                    k = f"{prefix}.residual_units.{block_idx}{suffix}"
                elif block_idx == 3:  # This is the activation
                    k = f"{prefix}.activation{suffix}"
                elif block_idx == 4:  # This is the final conv
                    k = f"{prefix}.conv{suffix}"
        elif "generator" in old_k:
            k = old_k.replace("generator", "decoder")

            # Handle prior_net -> prior_blocks conversion
            if "backbone.prior_net.0." in k:
                k = k.replace("backbone.prior_net.0.", "backbone.prior_blocks.0.")
            elif "backbone.prior_net.1." in k:
                k = k.replace("backbone.prior_net.1.", "backbone.prior_blocks.1.")

            # Handle post_net -> post_blocks conversion
            elif "backbone.post_net.0." in k:
                k = k.replace("backbone.post_net.0.", "backbone.post_blocks.0.")
            elif "backbone.post_net.1." in k:
                k = k.replace("backbone.post_net.1.", "backbone.post_blocks.1.")

            # Handle special cases for decoder.backbone.transformers
            elif "backbone.transformers" in k:
                if "c_attn" in k:
                    # split into 3 tensors
                    c_attn_weight = state_dict[old_k]
                    W_q, W_k, W_v = torch.chunk(c_attn_weight, chunks=3, dim=0)

                    n_heads = hf_model.config.num_attention_heads
                    W_q = permute_for_rope(W_q, n_heads, W_q.shape[0], W_q.shape[1])
                    W_k = permute_for_rope(W_k, n_heads, W_k.shape[0], W_k.shape[1])
                    k_mod = k.replace("att.", "self_attn.")
                    tensors[k_mod.replace("c_attn", "q_proj")] = W_q
                    tensors[k_mod.replace("c_attn", "k_proj")] = W_k
                    tensors[k_mod.replace("c_attn", "v_proj")] = W_v
                    continue  # Skip the rest of the loop for this key
                elif "c_proj" in k:
                    k = k.replace(".att.c_proj", ".self_attn.o_proj")
                elif "att_norm" in k:
                    k = k.replace("att_norm", "input_layernorm")
                elif "ffn_norm" in k:
                    k = k.replace("ffn_norm", "post_attention_layernorm")
        elif "SemanticEncoder_module" in old_k:
            k = old_k.replace("SemanticEncoder_module", "semantic_encoder")

            # Handle residual_blocks -> individual modules conversion
            if "residual_blocks.0." in k:
                k = k.replace("residual_blocks.0.", "act1.")
            elif "residual_blocks.1." in k:
                k = k.replace("residual_blocks.1.", "conv1.")
            elif "residual_blocks.2." in k:
                k = k.replace("residual_blocks.2.", "act2.")
            elif "residual_blocks.3." in k:
                k = k.replace("residual_blocks.3.", "conv2.")
        else:
            k = old_k

        # copy over to new state dict
        tensors[k] = state_dict[old_k]

    state_dict = tensors
    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    if len(extra_keys) != 0:
        raise ValueError(f"{len(extra_keys)} extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"{len(missing_keys)} missing keys found: {missing_keys}")
    hf_model.load_state_dict(state_dict, strict=True)
    n_params = param_count(hf_model)

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
