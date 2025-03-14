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

import argparse
import os
import re
import json
import torch

from transformers import (
    StyleTextToSpeech2Config,
    StyleTextToSpeech2Model,
    StyleTextToSpeech2Tokenizer,
    StyleTextToSpeech2Processor,
)
from transformers.utils.hub import get_file_from_repo


# fmt: off
TEXT_ENCODER_MAPPING = {
    r"^module\.(.*)":                                                          r"\1",
    r"cnn":                                                                    r"layers",
    r"embedding":                                                              r"embed_tokens",
    r"gamma":                                                                  r"weight", 
    r"beta":                                                                   r"bias",
    r"\.(\d+)\.0":                                                             r".\1.conv",
    r"\.(\d+)\.1":                                                             r".\1.norm",
    r"weight_g":                                                               r"parametrizations.weight.original0",
    r"weight_v":                                                               r"parametrizations.weight.original1",
}
# fmt: on


# fmt: off
PREDICTOR_MAPPING = {
    r'^text_encoder\.lstms\.(\d+)\.weight_(ih|hh)_l0(_reverse)?$':             lambda m: f'prosody_encoder.layers.{int(m.group(1))//2}.lstm.weight_{m.group(2)}_l0{m.group(3) or ""}',
    r'^text_encoder\.lstms\.(\d+)\.bias_(ih|hh)_l0(_reverse)?$':               lambda m: f'prosody_encoder.layers.{int(m.group(1))//2}.lstm.bias_{m.group(2)}_l0{m.group(3) or ""}',
    r'^text_encoder\.lstms\.(\d+)\.fc\.(weight|bias)$':                        lambda m: f'prosody_encoder.layers.{int(m.group(1))//2}.ada_layer_norm.proj.{m.group(2)}',
    
    r'^lstms\.(\d+)\.weight_(ih|hh)_l0(_reverse)?$':                           lambda m: f'layers.{int(m.group(1))//2}.lstm.weight_{m.group(2)}_l0{m.group(3) or ""}',
    r'^lstms\.(\d+)\.bias_(ih|hh)_l0(_reverse)?$':                             lambda m: f'layers.{int(m.group(1))//2}.lstm.bias_{m.group(2)}_l0{m.group(3) or ""}',
    r'^lstms\.(\d+)\.fc\.(weight|bias)$':                                      lambda m: f'layers.{int(m.group(1))//2}.ada_layer_norm.proj.{m.group(2)}',

    r'^lstm\.(.*)$':                                                           r'duration_projector.lstm.\1',
    r'^duration_proj\.linear_layer\.(.*)$':                                    r'duration_projector.duration_proj.\1',
    r'^duration_proj\.(.*)$':                                                  r'duration_projector.duration_proj.\1',

    r'^(F0|N)\.(\d+)\.conv1\.bias$':                                           lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_{int(m.group(2))+1}.conv1.bias',
    r'^(F0|N)\.(\d+)\.conv1\.weight_g$':                                       lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_{int(m.group(2))+1}.conv1.parametrizations.weight.original0',
    r'^(F0|N)\.(\d+)\.conv1\.weight_v$':                                       lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_{int(m.group(2))+1}.conv1.parametrizations.weight.original1',
    r'^(F0|N)\.(\d+)\.conv2\.bias$':                                           lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_{int(m.group(2))+1}.conv2.bias',
    r'^(F0|N)\.(\d+)\.conv2\.weight_g$':                                       lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_{int(m.group(2))+1}.conv2.parametrizations.weight.original0',
    r'^(F0|N)\.(\d+)\.conv2\.weight_v$':                                       lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_{int(m.group(2))+1}.conv2.parametrizations.weight.original1',
    r'^(F0|N)\.(\d+)\.norm1\.fc\.weight$':                                     lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_{int(m.group(2))+1}.norm1.proj.weight',
    r'^(F0|N)\.(\d+)\.norm1\.fc\.bias$':                                       lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_{int(m.group(2))+1}.norm1.proj.bias',
    r'^(F0|N)\.(\d+)\.norm2\.fc\.weight$':                                     lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_{int(m.group(2))+1}.norm2.proj.weight',
    r'^(F0|N)\.(\d+)\.norm2\.fc\.bias$':                                       lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_{int(m.group(2))+1}.norm2.proj.bias',
    r'^(F0|N)\.1\.conv1x1\.weight_g$':                                         lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_2.conv1_shortcut.parametrizations.weight.original0',
    r'^(F0|N)\.1\.conv1x1\.weight_v$':                                         lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_2.conv1_shortcut.parametrizations.weight.original1',
    r'^(F0|N)\.1\.pool\.bias$':                                                lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_2.pool.bias',
    r'^(F0|N)\.1\.pool\.weight_g$':                                            lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_2.pool.parametrizations.weight.original0',
    r'^(F0|N)\.1\.pool\.weight_v$':                                            lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.adain_res_1d_2.pool.parametrizations.weight.original1',
    r'^(F0|N)_proj\.weight$':                                                  lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.conv_out.weight',
    r'^(F0|N)_proj\.bias$':                                                    lambda m: f'prosody_predictor.{"pitch" if m.group(1)=="F0" else "energy"}_block.conv_out.bias',

    r'^shared\.weight_(ih|hh)_l0(_reverse)?$':                                 lambda m: f'prosody_predictor.lstm.weight_{m.group(1)}_l0{m.group(2) or ""}',
    r'^shared\.bias_(ih|hh)_l0(_reverse)?$':                                   lambda m: f'prosody_predictor.lstm.bias_{m.group(1)}_l0{m.group(2) or ""}',
}
# fmt: on


# fmt: off
DECODER_MAPPING = {
    r"^(generator\.[^.]+\.(?:\d+)?)\.weight_(g|v)$":                           lambda match: f"{match.group(1)}.parametrizations.weight.original{'0' if match.group(2)=='g' else '1'}",
    r"^(generator\.[^.]+\.(?:\d+)?)\.(bias)$":                                 lambda match: f"{match.group(1)}.{match.group(2)}",

    r"^generator\.noise_convs\.(\d+)\.(weight|bias)$":                         r"generator.layers.\1.noise_conv.\2",

    r"^generator\.noise_res\.(\d+)\.convs([12])\.(\d+)\.weight_(g|v)$":        lambda match: f"generator.layers.{match.group(1)}.noise_res.layers.{match.group(3)}.conv{match.group(2)}.parametrizations.weight.original{'0' if match.group(4)=='g' else '1'}",
    r"^generator\.noise_res\.(\d+)\.convs([12])\.(\d+)\.(bias)$":              r"generator.layers.\1.noise_res.layers.\3.conv\2.\4",
    r"^generator\.noise_res\.(\d+)\.adain([12])\.(\d+)\.fc\.(weight|bias)$":   r"generator.layers.\1.noise_res.layers.\3.norm\2.proj.\4",
    r"^generator\.noise_res\.(\d+)\.alpha([12])\.(\d+)$":                      r"generator.layers.\1.noise_res.layers.\3.alpha\2",

    r"^generator\.ups\.(\d+)\.parametrizations\.weight\.original([01])$":      lambda match: f"generator.layers.{match.group(1)}.up.parametrizations.weight.original{match.group(2)}",
    r"^generator\.ups\.(\d+)\.(bias)$":                                        r"generator.layers.\1.up.\2",

    r"^generator\.resblocks\.(\d+)\.convs([12])\.(\d+)\.weight_(g|v)$":        lambda match: f"generator.layers.{int(match.group(1)) // 3}.resblocks.{int(match.group(1)) % 3}.layers.{match.group(3)}.conv{match.group(2)}.parametrizations.weight.original{'0' if match.group(4)=='g' else '1'}",
    r"^generator\.resblocks\.(\d+)\.convs([12])\.(\d+)\.(bias)$":              lambda match: f"generator.layers.{int(match.group(1)) // 3}.resblocks.{int(match.group(1)) % 3}.layers.{match.group(3)}.conv{match.group(2)}.{match.group(4)}",
    r"^generator\.resblocks\.(\d+)\.adain([12])\.(\d+)\.fc\.(weight|bias)$":   lambda match: f"generator.layers.{int(match.group(1)) // 3}.resblocks.{int(match.group(1)) % 3}.layers.{match.group(3)}.norm{match.group(2)}.proj.{match.group(4)}",
    r"^generator\.resblocks\.(\d+)\.alpha([12])\.(\d+)$":                      lambda match: f"generator.layers.{int(match.group(1)) // 3}.resblocks.{int(match.group(1)) % 3}.layers.{match.group(3)}.alpha{match.group(2)}",

    r"^generator\.conv_post\.(bias)$":                                         r"generator.conv_post.\1",
    r"^generator\.conv_post\.weight_(g|v)$":                                   lambda match: f"generator.conv_post.parametrizations.weight.original{'0' if match.group(1)=='g' else '1'}",

    r"^(F0|N)_conv\.(weight_g|weight_v|bias)$":                                lambda match: f"{'pitch' if match.group(1)=='F0' else 'energy'}_conv.{'parametrizations.weight.original0' if match.group(2)=='weight_g' else 'parametrizations.weight.original1' if match.group(2)=='weight_v' else 'bias'}",

    r"^encode\.(conv1|conv2)\.(bias|weight_g|weight_v)$":                      lambda match: f"acoustic_encoder.{match.group(1)}.{'parametrizations.weight.original0' if match.group(2)=='weight_g' else 'parametrizations.weight.original1' if match.group(2)=='weight_v' else 'bias'}",
    r"^encode\.(norm1|norm2)\.fc\.(weight|bias)$":                             r"acoustic_encoder.\1.proj.\2",
    r"^encode\.conv1x1\.(weight_g|weight_v)$":                                 lambda match: f"acoustic_encoder.conv1_shortcut.parametrizations.weight.original{'0' if match.group(1)=='weight_g' else '1'}",

    r"^asr_res\.0\.(bias|weight_g|weight_v)$":                                 lambda match: f"acoustic_residual.{'parametrizations.weight.original0' if match.group(1)=='weight_g' else 'parametrizations.weight.original1' if match.group(1)=='weight_v' else 'bias'}",

    r"^decode\.(\d+)\.(conv1|conv2)\.(bias|weight_g|weight_v)$":               lambda match: f"decoder.{match.group(1)}.{match.group(2)}.{'parametrizations.weight.original0' if match.group(3)=='weight_g' else 'parametrizations.weight.original1' if match.group(3)=='weight_v' else 'bias'}",
    r"^decode\.(\d+)\.(norm1|norm2)\.fc\.(weight|bias)$":                      r"decoder.\1.\2.proj.\3",
    r"^decode\.(\d+)\.conv1x1\.(weight_g|weight_v)$":                          lambda match: f"decoder.{match.group(1)}.conv1_shortcut.parametrizations.weight.original{'0' if match.group(2)=='weight_g' else '1'}",
    r"^decode\.3\.pool\.(bias|weight_g|weight_v)$":                            lambda match: f"decoder.3.pool.{'parametrizations.weight.original0' if match.group(1)=='weight_g' else 'parametrizations.weight.original1' if match.group(1)=='weight_v' else 'bias'}",
}
# fmt: on


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def convert_shape(value):
    if value.shape == (1, 1, 1):
        return value
    elif value.shape[-1] == 1 and value.shape[0] == 1 and len(value.shape) > 1:
        return value.squeeze()
    else:
        return value


def write_model(
    input_path_or_repo,
    model_name,
    output_dir,
    safe_serialization=True,
):
    print("Converting the model.")
    os.makedirs(output_dir, exist_ok=True)

    config = StyleTextToSpeech2Config()
    model_path = get_file_from_repo(
            input_path_or_repo,
            model_name,
    )
    print(f"Fetching all parameters from the checkpoint at {model_path}...")
    loaded = torch.load(model_path, map_location="cpu")

    print("Converting model...")
    state_dict = {}

    # -----------------------
    # convert parameter names
    # -----------------------

    # acoustic text encoder
    state_dict.update({f"acoustic_text_encoder.{convert_key(key, TEXT_ENCODER_MAPPING)}": value for key, value in loaded["text_encoder"].items()})
    
    # predictor
    state_dict.update({f"predictor.{convert_key(key.replace('module.', ''), PREDICTOR_MAPPING)}": value for key, value in loaded["predictor"].items()})
    for key, value in loaded["bert"].items():
        key = key.replace('module.', '')
        corresponding_key = f"predictor.prosodic_text_encoder.bert_model.{key}"
        state_dict[corresponding_key] = value
    for key, value in loaded["bert_encoder"].items():
        key = key.replace('module.', '')
        corresponding_key = f"predictor.prosodic_text_encoder.proj_out.{key}"
        state_dict[corresponding_key] = value

    # decoder
    state_dict.update(
        {f"decoder.{convert_key(key.replace('module.', ''), DECODER_MAPPING)}": convert_shape(value) for key, value in loaded["decoder"].items()}
    )

    # -------------------------
    # load the weights and save
    # -------------------------

    model = StyleTextToSpeech2Model(config)
    model.load_state_dict(state_dict)

    print("Saving the model...")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    print(f"Model saved at {output_dir}!")


def write_tokenizer(output_dir: str, vocab_file: str):
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    # pad token
    vocab["$"] = 0  

    vocab_file = os.path.join(output_dir, "vocab.json")
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    tokenizer = StyleTextToSpeech2Tokenizer(vocab_file=vocab_file)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved at {output_dir}!")


def write_processor(
    output_dir: str,
    voice_presets_path_or_repo: str,
    voice_to_path_path: str,
):
    """
    Args:
        output_dir: str,
            Location to write HF model and tokenizer
        voice_presets_path_or_repo: str,
            Path or repo that will be the base diretory of paths in voice_to_path_path
        voice_to_path_path: str,
            Path to voice to path mapping is the voice_presets_path_or_repo directory.
            Should be a json file with the following format:
            {
                "voice_name": "path_to_voice_preset"
            }
    """
    print(f"Loading voice presets from {voice_presets_path_or_repo}...")
    with open(voice_to_path_path, "r", encoding="utf-8") as f:
        voice_to_path = json.load(f)

    voice_presets_config = {
        "path_or_repo": voice_presets_path_or_repo,
        "voice_to_path": voice_to_path,
    }
    processor = StyleTextToSpeech2Processor(
        tokenizer=StyleTextToSpeech2Tokenizer.from_pretrained(output_dir), 
        voice_presets_config=voice_presets_config,
    )

    processor.save_pretrained(output_dir)
    print(f"Processor saved at {output_dir}!")


def main():
    parser = argparse.ArgumentParser(description="Convert Kokoro weights to HuggingFace format")
    parser.add_argument(
        "--input_path_or_repo",
        type=str,
        required=True,
        help="Path or repo containing Kokoro weights",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model in input_path_or_repo",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", action="store_true", default=True, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="Path to vocab file",
    )
    parser.add_argument(
        "--voice_presets_path_or_repo",
        type=str,
        required=True,
        help="Path or repo that will be the base diretory of paths in voice_to_path_path",
    )
    parser.add_argument(
        "--voice_to_path_path",
        type=str,
        required=True,
        help="Path to voice to path mapping is the voice_presets_path_or_repo directory.",
    )
    args = parser.parse_args()

    write_model(
        args.input_path_or_repo,
        args.model_name,
        output_dir=args.output_dir, 
        safe_serialization=args.safe_serialization,
    )
     
    write_tokenizer(
        args.output_dir,
        args.vocab_file,
    )

    write_processor(
        args.output_dir,
        args.voice_presets_path_or_repo,
        args.voice_to_path_path,
    )


if __name__ == "__main__":
    main()
