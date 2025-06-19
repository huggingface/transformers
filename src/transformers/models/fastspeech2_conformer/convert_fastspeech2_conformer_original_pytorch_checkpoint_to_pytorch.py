# coding=utf-8
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
"""Convert FastSpeech2Conformer checkpoint."""

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml

from transformers import (
    FastSpeech2ConformerConfig,
    FastSpeech2ConformerModel,
    FastSpeech2ConformerTokenizer,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")

CONFIG_MAPPING = {
    "adim": "hidden_size",
    "aheads": "num_attention_heads",
    "conformer_dec_kernel_size": "decoder_kernel_size",
    "conformer_enc_kernel_size": "encoder_kernel_size",
    "decoder_normalize_before": "decoder_normalize_before",
    "dlayers": "decoder_layers",
    "dunits": "decoder_linear_units",
    "duration_predictor_chans": "duration_predictor_channels",
    "duration_predictor_kernel_size": "duration_predictor_kernel_size",
    "duration_predictor_layers": "duration_predictor_layers",
    "elayers": "encoder_layers",
    "encoder_normalize_before": "encoder_normalize_before",
    "energy_embed_dropout": "energy_embed_dropout",
    "energy_embed_kernel_size": "energy_embed_kernel_size",
    "energy_predictor_chans": "energy_predictor_channels",
    "energy_predictor_dropout": "energy_predictor_dropout",
    "energy_predictor_kernel_size": "energy_predictor_kernel_size",
    "energy_predictor_layers": "energy_predictor_layers",
    "eunits": "encoder_linear_units",
    "pitch_embed_dropout": "pitch_embed_dropout",
    "pitch_embed_kernel_size": "pitch_embed_kernel_size",
    "pitch_predictor_chans": "pitch_predictor_channels",
    "pitch_predictor_dropout": "pitch_predictor_dropout",
    "pitch_predictor_kernel_size": "pitch_predictor_kernel_size",
    "pitch_predictor_layers": "pitch_predictor_layers",
    "positionwise_conv_kernel_size": "positionwise_conv_kernel_size",
    "postnet_chans": "speech_decoder_postnet_units",
    "postnet_filts": "speech_decoder_postnet_kernel",
    "postnet_layers": "speech_decoder_postnet_layers",
    "reduction_factor": "reduction_factor",
    "stop_gradient_from_energy_predictor": "stop_gradient_from_energy_predictor",
    "stop_gradient_from_pitch_predictor": "stop_gradient_from_pitch_predictor",
    "transformer_dec_attn_dropout_rate": "decoder_attention_dropout_rate",
    "transformer_dec_dropout_rate": "decoder_dropout_rate",
    "transformer_dec_positional_dropout_rate": "decoder_positional_dropout_rate",
    "transformer_enc_attn_dropout_rate": "encoder_attention_dropout_rate",
    "transformer_enc_dropout_rate": "encoder_dropout_rate",
    "transformer_enc_positional_dropout_rate": "encoder_positional_dropout_rate",
    "use_cnn_in_conformer": "use_cnn_in_conformer",
    "use_macaron_style_in_conformer": "use_macaron_style_in_conformer",
    "use_masking": "use_masking",
    "use_weighted_masking": "use_weighted_masking",
    "idim": "input_dim",
    "odim": "num_mel_bins",
    "spk_embed_dim": "speaker_embed_dim",
    "langs": "num_languages",
    "spks": "num_speakers",
}


def remap_model_yaml_config(yaml_config_path):
    with Path(yaml_config_path).open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
        args = argparse.Namespace(**args)

    remapped_config = {}

    model_params = args.tts_conf["text2mel_params"]
    # espnet_config_key -> hf_config_key, any keys not included are ignored
    for espnet_config_key, hf_config_key in CONFIG_MAPPING.items():
        if espnet_config_key in model_params:
            remapped_config[hf_config_key] = model_params[espnet_config_key]

    return remapped_config, args.g2p, args.token_list


def convert_espnet_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if "tts.generator.text2mel." in key:
            new_key = key.replace("tts.generator.text2mel.", "")
            if "postnet" in key:
                new_key = new_key.replace("postnet.postnet", "speech_decoder_postnet.layers")
                new_key = new_key.replace(".0.weight", ".conv.weight")
                new_key = new_key.replace(".1.weight", ".batch_norm.weight")
                new_key = new_key.replace(".1.bias", ".batch_norm.bias")
                new_key = new_key.replace(".1.running_mean", ".batch_norm.running_mean")
                new_key = new_key.replace(".1.running_var", ".batch_norm.running_var")
                new_key = new_key.replace(".1.num_batches_tracked", ".batch_norm.num_batches_tracked")
            if "feat_out" in key:
                if "weight" in key:
                    new_key = "speech_decoder_postnet.feat_out.weight"
                if "bias" in key:
                    new_key = "speech_decoder_postnet.feat_out.bias"
            if "encoder.embed.0.weight" in key:
                new_key = new_key.replace("0.", "")
            if "w_1" in key:
                new_key = new_key.replace("w_1", "conv1")
            if "w_2" in key:
                new_key = new_key.replace("w_2", "conv2")
            if "predictor.conv" in key:
                new_key = new_key.replace(".conv", ".conv_layers")
                pattern = r"(\d)\.(\d)"
                replacement = (
                    r"\1.conv" if ("2.weight" not in new_key) and ("2.bias" not in new_key) else r"\1.layer_norm"
                )
                new_key = re.sub(pattern, replacement, new_key)
            if "pitch_embed" in key or "energy_embed" in key:
                new_key = new_key.replace("0", "conv")
            if "encoders" in key:
                new_key = new_key.replace("encoders", "conformer_layers")
                new_key = new_key.replace("norm_final", "final_layer_norm")
                new_key = new_key.replace("norm_mha", "self_attn_layer_norm")
                new_key = new_key.replace("norm_ff_macaron", "ff_macaron_layer_norm")
                new_key = new_key.replace("norm_ff", "ff_layer_norm")
                new_key = new_key.replace("norm_conv", "conv_layer_norm")
            if "lid_emb" in key:
                new_key = new_key.replace("lid_emb", "language_id_embedding")
            if "sid_emb" in key:
                new_key = new_key.replace("sid_emb", "speaker_id_embedding")

            new_state_dict[new_key] = state_dict[key]

    return new_state_dict


@torch.no_grad()
def convert_FastSpeech2ConformerModel_checkpoint(
    checkpoint_path,
    yaml_config_path,
    pytorch_dump_folder_path,
    repo_id=None,
):
    model_params, tokenizer_name, vocab = remap_model_yaml_config(yaml_config_path)
    config = FastSpeech2ConformerConfig(**model_params)

    # Prepare the model
    model = FastSpeech2ConformerModel(config)

    espnet_checkpoint = torch.load(checkpoint_path, weights_only=True)
    hf_compatible_state_dict = convert_espnet_state_dict_to_hf(espnet_checkpoint)

    model.load_state_dict(hf_compatible_state_dict)

    model.save_pretrained(pytorch_dump_folder_path)

    # Prepare the tokenizer
    with TemporaryDirectory() as tempdir:
        vocab = {token: id for id, token in enumerate(vocab)}
        vocab_file = Path(tempdir) / "vocab.json"
        with open(vocab_file, "w") as f:
            json.dump(vocab, f)
        should_strip_spaces = "no_space" in tokenizer_name
        tokenizer = FastSpeech2ConformerTokenizer(str(vocab_file), should_strip_spaces=should_strip_spaces)

    tokenizer.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument(
        "--yaml_config_path", required=True, default=None, type=str, help="Path to config.yaml of model to convert"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_FastSpeech2ConformerModel_checkpoint(
        args.checkpoint_path,
        args.yaml_config_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
