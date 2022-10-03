# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Convert Whisper checkpoints from the original repository. URL: https://github.com/openai/whisper"""

import argparse
import hashlib
import io
import os
import urllib
import warnings

import torch
from tqdm import tqdm

from transformers import WhisperConfig, WhisperForConditionalGeneration


def remove_ignore_keys_(state_dict):
    ignore_keys = ["layers", "blocks"]
    for k in ignore_keys:
        state_dict.pop(k, None)


WHISPER_MAPPING = {
    "blocks": "layers",
    "mlp.0": "fc1",
    "mlp.2": "fc2",
    "mlp_ln": "final_layer_norm",
    "blocks": "layers",
    ".attn.query": ".self_attn.q_proj",
    ".attn.key": ".self_attn.k_proj",
    ".attn.value": ".self_attn.v_proj",
    ".attn_ln": ".self_attn_layer_norm",
    ".attn.out": ".self_attn.out_proj",
    ".cross_attn.query": ".encoder_attn.q_proj",
    ".cross_attn.key": ".encoder_attn.k_proj",
    ".cross_attn.value": ".encoder_attn.v_proj",
    ".cross_attn_ln": ".encoder_attn_layer_norm",
    ".cross_attn.out": ".encoder_attn.out_proj",
    "decoder.ln.": "decoder.layer_norm.",
    "encoder.ln.": "encoder.layer_norm.",
    "token_embedding": "embed_tokens",
    "encoder.positional_embedding": "encoder.embed_positions.weight",
    "decoder.positional_embedding": "decoder.embed_positions.weight",
    "ln_post": "layer_norm",
}


def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for old_key, new_key in WHISPER_MAPPING.items():
            if old_key in key:
                new_key = new_key.replace(old_key, new_key)

        print(f"{key} -> {new_key}")

        s_dict[new_key] = s_dict.pop(key)
    return s_dict


def convert_openai_whisper_to_tfms(checkpoint_name, pytorch_dump_folder_path, checkpoint_path="weights"):
    full_path = os.path.join(os.getcwd(), checkpoint_path)
    if not os.path.isdir(os.path.join(full_path)):
        os.makedirs(full_path, exist_ok=True)
        try:
            _, checkpoint_path = _download(_MODELS[checkpoint_name], full_path)
        except KeyError:
            print("The original checkpoint should be in _MODELS ")

    print(f"Loading model from : {full_path}/{checkpoint_name}")
    original_checkpoint = torch.load(os.path.join(full_path, checkpoint_name) + ".pt", map_location="cpu")
    dimensions = original_checkpoint["dims"]
    state_dict = original_checkpoint["model_state_dict"]

    remove_ignore_keys_(state_dict)
    rename_keys(state_dict)

    config = WhisperConfig(
        vocab_size=dimensions["n_vocab"],
        num_mel_bins=dimensions["n_mels"],
        d_model=dimensions["n_audio_state"],
        max_target_positions=dimensions["n_text_ctx"],
        encoder_layers=dimensions["n_audio_layer"],
        encoder_attention_heads=dimensions["n_audio_head"],
        decoder_layers=dimensions["n_text_layer"],
        decoder_attention_heads=dimensions["n_text_head"],
        max_source_positions=dimensions["n_audio_ctx"],
        decoder_ffn_dim=4 * dimensions["n_audio_state"],
        encoder_ffn_dim=4 * dimensions["n_audio_state"],
    )

    model = WhisperForConditionalGeneration(config)
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 and not set(missing) <= set(
        [
            "encoder.embed_positions.weights",
            "decoder.embed_positions.weights",
        ]
    ):
        raise ValueError(
            "Only `encoder.embed_positions.weights` and `decoder.embed_positions.weights`  are allowed to be missing,"
            f" but all the following weights are missing {missing}"
        )

    model.save_pretrained(os.path.join(pytorch_dump_folder_path, checkpoint_name))


_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
}


def _download(url: str, root: str) -> bytes:
    # Copied from whisper's original codebase
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        model_bytes = open(download_target, "rb").read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return None, download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True, unit_divisor=1024
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return model_bytes, download_target


def convert_every_model(save_dir="whisper"):
    layers = [4, 6, 12, 24, 32, 4, 6, 12, 24]
    width = [384, 512, 768, 1024, 1280, 384, 512, 768, 1024]
    heads = [6, 8, 12, 16, 20, 6, 8, 12, 16]
    name = ["tiny", "base", "small", "medium", "large", "tiny.en", "base.en", "small.en", "medium.en"]
    for l, w, h, n in zip(layers, width, heads, name):

        config = WhisperConfig(
            vocab_size=51865,
            encoder_layers=l,
            encoder_attention_heads=h,
            decoder_attention_heads=h,
            decoder_layers=l,
            d_model=w,
            decoder_ffn_dim=4 * w,
            encoder_ffn_dim=4 * w,
        )
        model = WhisperForConditionalGeneration(config)

        model_bytes, _ = _download(_MODELS[n], "original-weights")
        with io.BytesIO(model_bytes) as fp:
            original = torch.load(fp, map_location="cpu")["model_state_dict"]

        new = rename_keys(original.copy())

        missing, unexpected = model.load_state_dict(new, strict=False)
        if missing == []:
            print("succesfully loaded")
            model.save_pretrained(f"{save_dir}/{n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--original_name", type=str, help="Path to the fairseq model (.pt) file.")
    parser.add_argument(
        "--pytorch_dump_folder_path", default="whisper-converted", type=str, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()

    convert_openai_whisper_to_tfms(args.original_name, args.pytorch_dump_folder_path)
