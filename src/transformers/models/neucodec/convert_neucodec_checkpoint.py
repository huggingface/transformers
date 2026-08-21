# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import re
import tempfile

import torch
import torch.nn as nn
from huggingface_hub import create_branch, upload_folder

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    NeuCodecConfig,
    NeuCodecFeatureExtractor,
    NeuCodecModel,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.neucodec")

# Original key names: https://github.com/neuphonic/neucodec/blob/main/neucodec/model.py
# NeuCodec forked its module naming directly from xcodec2 (https://huggingface.co/HKUSTAudio/xcodec2), so the
# mapping below mirrors `xcodec2/convert_xcodec2_checkpoint.py`. Mappings are applied sequentially per key; order
# matters.
# fmt: off
STATE_DICT_MAPPING = {
    # ── Semantic encoder (semantic_model -> semantic_encoder) ──
    r"^semantic_model\.":                                                           r"semantic_encoder.",

    # ── Acoustic encoder (CodecEnc -> acoustic_encoder) ──
    r"^CodecEnc\.conv_blocks\.0\.":                                                 r"acoustic_encoder.conv1.",
    r"^CodecEnc\.conv_final_block\.0\.":                                            r"acoustic_encoder.snake1.",
    r"^CodecEnc\.conv_final_block\.1\.":                                            r"acoustic_encoder.conv2.",
    r"^CodecEnc\.conv_blocks\.(\d+)\.":                                             r"acoustic_encoder.block.SHIFT_DOWN.\1.",
    r"^CodecEnc\.":                                                                 r"acoustic_encoder.",
    # -- EncoderBlock: block.{0,1,2} -> res_unit{1,2,3}, block.3 -> snake1, block.4 -> conv1
    # NOTE: These must come BEFORE ResidualUnit mappings so keys are transformed first
    r"(\.block\.\d+)\.block\.0\.":                                                 r"\1.res_unit1.",
    r"(\.block\.\d+)\.block\.1\.":                                                 r"\1.res_unit2.",
    r"(\.block\.\d+)\.block\.2\.":                                                 r"\1.res_unit3.",
    r"(\.block\.\d+)\.block\.3\.":                                                 r"\1.snake1.",
    r"(\.block\.\d+)\.block\.4\.":                                                 r"\1.conv1.",
    # -- ResidualUnit: block.{0,1,2,3} -> {snake1,conv1,snake2,conv2}
    r"(\.res_unit\d+)\.block\.0\.":                                                 r"\1.snake1.",
    r"(\.res_unit\d+)\.block\.1\.":                                                 r"\1.conv1.",
    r"(\.res_unit\d+)\.block\.2\.":                                                 r"\1.snake2.",
    r"(\.res_unit\d+)\.block\.3\.":                                                 r"\1.conv2.",
    # -- ResidualUnit nested structure: res_unit{1,2,3,4} -> {snake1,conv1,snake2,conv2}
    r"(\.res_unit\d+)\.res_unit1\.":                                                r"\1.snake1.",
    r"(\.res_unit\d+)\.res_unit2\.":                                                r"\1.conv1.",
    r"(\.res_unit\d+)\.res_unit3\.":                                                r"\1.snake2.",
    r"(\.res_unit\d+)\.res_unit4\.":                                                r"\1.conv2.",
    # -- DownSample1d: lowpass.filter -> filter (LowPassFilter1d inlined)
    r"\.downsample\.lowpass\.filter":                                               r".downsample.filter",

    # ── Quantizer: lives on main model, not inside decoder ──
    r"^generator\.quantizer\.layers\.0\.":                                          r"quantizer.quantizer.",
    r"^generator\.quantizer\.":                                                     r"quantizer.",

    # ── Decoder (generator -> acoustic_decoder) ──
    # -- Handle backbone components: now directly in decoder (no backbone. prefix)
    r"^generator\.backbone\.prior_net\.":                                          r"acoustic_decoder.prior_net.",
    r"^generator\.backbone\.post_net\.":                                           r"acoustic_decoder.post_net.",
    r"^generator\.backbone\.final_layer_norm\.":                                   r"acoustic_decoder.norm.",
    r"^generator\.backbone\.":                                                      r"acoustic_decoder.",
    # -- General generator mapping
    r"^generator\.":                                                                r"acoustic_decoder.",
    # -- ISTFT head: out -> linear
    r"acoustic_decoder\.head\.out\.":                                              r"acoustic_decoder.head.linear.",
    # -- Transformer layers: transformers -> layers
    r"\.transformers\.":                                                            r".layers.",
    r"\.att\.c_proj\.":                                                             r".self_attn.o_proj.",
    r"\.att_norm\.":                                                                r".input_layernorm.",
    r"\.ffn_norm\.":                                                                r".post_attention_layernorm.",

    # ── Semantic adapter (SemanticEncoder_module -> semantic_adapter) ──
    r"^SemanticEncoder_module\.":                                                   r"semantic_adapter.",
    r"semantic_adapter\.initial_conv\.":                                            r"semantic_adapter.conv1.",
    r"semantic_adapter\.final_conv\.":                                              r"semantic_adapter.conv4.",
    r"semantic_adapter\.residual_blocks\.0\.":                                      r"semantic_adapter.act1.",
    r"semantic_adapter\.residual_blocks\.1\.":                                      r"semantic_adapter.conv2.",
    r"semantic_adapter\.residual_blocks\.2\.":                                      r"semantic_adapter.act2.",
    r"semantic_adapter\.residual_blocks\.3\.":                                      r"semantic_adapter.conv3.",

    # ── Linear layers (fc_prior -> fc_encoder, fc_post_a -> acoustic_decoder.fc) ──
    r"^fc_prior\.":                                                                 r"fc_encoder.",
    r"^fc_post_a\.":                                                                r"acoustic_decoder.fc.",
}
# fmt: on

# Training-only keys present in the released checkpoint that have no counterpart in the inference-only
# HF port: a semantic reconstruction auxiliary head (`fc_post_s`, `SemanticDecoder`). See
# https://github.com/neuphonic/neucodec/blob/main/neucodec/model.py#L69 (`ignore_keys` for `neuphonic/neucodec`).
IGNORED_KEY_SUBSTRINGS = ["fc_post_s", "SemanticDecoder"]


def permute_for_rope(input_tensor, n_heads, dim1, dim2):
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


def _is_unused_semantic_layer(key, num_layers):
    """Check if a key belongs to a semantic encoder layer beyond the ones we keep."""
    match = re.search(r"semantic_encoder\.encoder\.layers\.(\d+)\.", key)
    return match is not None and int(match.group(1)) >= num_layers


def convert_state_dict(state_dict, hf_model):
    new_state_dict = {}

    state_dict = {k: v for k, v in state_dict.items() if not any(s in k for s in IGNORED_KEY_SUBSTRINGS)}

    for old_key, tensor in state_dict.items():
        # Special case: c_attn is a fused QKV projection that must be split into 3 tensors
        if ".att.c_attn." in old_key:
            w_q, w_k, w_v = torch.chunk(tensor, chunks=3, dim=0)
            w_q = permute_for_rope(w_q, hf_model.config.num_attention_heads, w_q.shape[0], w_q.shape[1])
            w_k = permute_for_rope(w_k, hf_model.config.num_attention_heads, w_k.shape[0], w_k.shape[1])
            base_key = map_old_key_to_new(old_key.replace(".att.c_attn.", ".self_attn.PROJ."))
            new_state_dict[base_key.replace("PROJ", "q_proj")] = w_q
            new_state_dict[base_key.replace("PROJ", "k_proj")] = w_k
            new_state_dict[base_key.replace("PROJ", "v_proj")] = w_v
            continue

        new_key = map_old_key_to_new(old_key)

        new_state_dict[new_key] = tensor
        if old_key != new_key:
            logger.debug(f"Converted: {old_key} -> {new_key}")

    state_dict = new_state_dict

    # Filter out non-persistent buffers (recomputed in _init_weights), unused semantic encoder layers (original
    # has 24 layers, we only use 16), and stale plain `.weight` entries left over for weight-normalized conv
    # layers whose checkpoint also contains the `weight_g`/`weight_v` reparametrization (the `.weight` there is a
    # redundant cached value, not a real parameter to load).
    num_semantic_layers = hf_model.config.semantic_model_config.num_hidden_layers
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not (
            k.endswith("upsample.filter")
            or k.endswith("downsample.filter")
            or k.endswith(".window")
            or k.endswith(".levels")
            or k.endswith(".basis")
            or k.endswith(".codebook")
            or _is_unused_semantic_layer(k, num_semantic_layers)
            or (k.endswith(".weight") and f"{k}_g" in state_dict)
        )
    }

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


def apply_weight_norm(model):
    weight_norm = nn.utils.weight_norm

    weight_norm(model.acoustic_encoder.conv1)
    for encoder_block in model.acoustic_encoder.block:
        weight_norm(encoder_block.res_unit1.conv1)
        weight_norm(encoder_block.res_unit1.conv2)
        weight_norm(encoder_block.res_unit2.conv1)
        weight_norm(encoder_block.res_unit2.conv2)
        weight_norm(encoder_block.res_unit3.conv1)
        weight_norm(encoder_block.res_unit3.conv2)
        weight_norm(encoder_block.conv1)
    weight_norm(model.acoustic_encoder.conv2)

    # NeuCodec's `CodecDecoderVocos.apply_weight_norm` (unlike xcodec2's) weight-norms every Conv1d in the
    # backbone, including `embed` and the pre/post ResNet blocks: https://github.com/neuphonic/neucodec/blob/main/neucodec/codec_decoder_vocos.py
    weight_norm(model.acoustic_decoder.embed)
    for resnet_block in [*model.acoustic_decoder.prior_net, *model.acoustic_decoder.post_net]:
        weight_norm(resnet_block.conv1)
        weight_norm(resnet_block.conv2)


def remove_weight_norm(model):
    param_remove = nn.utils.remove_weight_norm

    param_remove(model.acoustic_encoder.conv1)
    for encoder_block in model.acoustic_encoder.block:
        param_remove(encoder_block.res_unit1.conv1)
        param_remove(encoder_block.res_unit1.conv2)
        param_remove(encoder_block.res_unit2.conv1)
        param_remove(encoder_block.res_unit2.conv2)
        param_remove(encoder_block.res_unit3.conv1)
        param_remove(encoder_block.res_unit3.conv2)
        param_remove(encoder_block.conv1)
    param_remove(model.acoustic_encoder.conv2)

    param_remove(model.acoustic_decoder.embed)
    for resnet_block in [*model.acoustic_decoder.prior_net, *model.acoustic_decoder.post_net]:
        param_remove(resnet_block.conv1)
        param_remove(resnet_block.conv2)


@torch.no_grad()
def convert_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path=None,
    repo_id=None,
    repo_subfolder=None,
    revision=None,
):
    # NeuCodec ships a bare `pytorch_model.bin` state dict with no accompanying `config.json` (only a `meta.yaml`
    # tracking downloads), so the architecture hyperparameters below are hardcoded from
    # https://github.com/neuphonic/neucodec/blob/main/neucodec/model.py and
    # https://github.com/neuphonic/neucodec/blob/main/neucodec/codec_encoder.py
    config = NeuCodecConfig(
        encoder_hidden_size=48,
        hidden_size=1024,
        output_sampling_rate=24000,
        semantic_model_config=AutoConfig.from_pretrained("facebook/w2v-bert-2.0", num_hidden_layers=16),
    )

    if not torch.cuda.is_available():
        raise ValueError("Run this script on a machine with a GPU for weight norm layers to be correctly copied.")
    torch_device = "cuda"
    model = NeuCodecModel(config).to(torch_device).eval()

    original_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Unlike xcodec2, NeuCodec's `pytorch_model.bin` does not bundle semantic-encoder weights: at load time the
    # reference implementation always fetches a frozen `facebook/w2v-bert-2.0` from the Hub instead
    # (https://github.com/neuphonic/neucodec/blob/main/neucodec/model.py#L28). Merge those weights in under the
    # `semantic_model.` prefix so `STATE_DICT_MAPPING` can route them like any other key.
    semantic_model = AutoModel.from_pretrained("facebook/w2v-bert-2.0")
    original_checkpoint.update({f"semantic_model.{k}": v for k, v in semantic_model.state_dict().items()})

    apply_weight_norm(model)
    model = convert_state_dict(original_checkpoint, model)
    remove_weight_norm(model)

    dac_fe = AutoFeatureExtractor.from_pretrained("descript/dac_16khz")
    semantic_fe = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    if semantic_fe.sampling_rate != dac_fe.sampling_rate:
        raise ValueError(
            f"Sampling rates for DAC and semantic feature extractor must match, got {dac_fe.sampling_rate} and {semantic_fe.sampling_rate}."
        )
    feature_extractor = NeuCodecFeatureExtractor(
        feature_size=semantic_fe.feature_size,
        sampling_rate=semantic_fe.sampling_rate,
        padding_value=semantic_fe.padding_value,
        hop_length=dac_fe.hop_length,
    )

    if pytorch_dump_folder_path is not None:
        model.save_pretrained(pytorch_dump_folder_path)
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        if revision:
            # `upload_folder`/`create_commit` (unlike `push_to_hub`) don't create a missing branch on their own.
            create_branch(repo_id=repo_id, branch=revision, exist_ok=True)
        if repo_subfolder:
            # `push_to_hub` always writes to the repo root, so for a subfolder destination (e.g. keeping the
            # original, non-HF checkpoint at the repo root while adding a converted copy alongside it) save
            # locally first and upload that folder to `repo_subfolder`.
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                feature_extractor.save_pretrained(tmp_dir)
                upload_folder(repo_id=repo_id, folder_path=tmp_dir, path_in_repo=repo_subfolder, revision=revision)
        else:
            feature_extractor.push_to_hub(repo_id, revision=revision)
            model.push_to_hub(repo_id, revision=revision)


"""
NeuCodec extends xcodec2 (see modular_neucodec.py): same encoder/decoder architecture and the same 16kHz-input
Wav2Vec2-BERT semantic encoder truncated to 16 layers, but the decoder synthesizes audio at 24kHz (hop_length=480)
instead of 16kHz (hop_length=320). See https://huggingface.co/neuphonic/neucodec.

Conversion example usage:
```
# download model weights
wget https://huggingface.co/neuphonic/neucodec/resolve/main/pytorch_model.bin -P /raid/eric/neucodec_original

# run conversion, first pushing to a test branch to sanity-check the result before touching `main`
python src/transformers/models/neucodec/convert_neucodec_checkpoint.py \
    --checkpoint_path /raid/eric/neucodec_original/pytorch_model.bin \
    --push_to_hub neuphonic/neucodec \
    --revision test-hf-conversion

# once verified, push to the repo root on `main` (config.json / model.safetensors sit alongside the existing
# pytorch_model.bin / meta.yaml without conflict, see `--repo_subfolder` for isolating them in a subfolder instead)
python src/transformers/models/neucodec/convert_neucodec_checkpoint.py \
    --checkpoint_path /raid/eric/neucodec_original/pytorch_model.bin \
    --push_to_hub neuphonic/neucodec
```

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )
    parser.add_argument(
        "--repo_subfolder",
        default=None,
        type=str,
        help="Subfolder of `--push_to_hub` to upload the converted model to, leaving the repo root untouched "
        "(e.g. the original, non-HF checkpoint). If unset, uploads to the repo root as usual.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        type=str,
        help="Branch of `--push_to_hub` to upload the converted model to (created if it doesn't exist yet), e.g. "
        "for testing on a branch before merging to `main`. If unset, uploads to `main`.",
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
        args.repo_subfolder,
        args.revision,
    )
