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
"""Convert VITS checkpoint."""

import argparse
import json

import torch
from huggingface_hub import hf_hub_download

from transformers import VitsConfig, logging

# TODO: change once added
from transformers.models.vits.modeling_vits import VitsDiscriminator


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.vits")


MAPPING = {
    "conv_post": "final_conv",
}
TOP_LEVEL_KEYS = []
IGNORE_KEYS = []


@torch.no_grad()
def convert_checkpoint(
    pytorch_dump_folder_path,
    checkpoint_path=None,
    config_path=None,
    vocab_path=None,
    language=None,
    num_speakers=None,
    sampling_rate=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = VitsConfig.from_pretrained(config_path)
    else:
        config = VitsConfig()

    if num_speakers:
        config.num_speakers = num_speakers
        config.speaker_embedding_size = 256

    if sampling_rate:
        config.sampling_rate = sampling_rate

    if checkpoint_path is None:
        logger.info(f"***Converting model: facebook/mms-tts {language}***")

        vocab_path = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="vocab.txt",
            subfolder=f"models/{language}",
        )
        config_file = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="config.json",
            subfolder=f"models/{language}",
        )
        checkpoint_path = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="D_100000.pth",
            subfolder=f"models/{language}",
        )

        with open(config_file, "r") as f:
            data = f.read()
            hps = json.loads(data)

        is_uroman = hps["data"]["training_files"].split(".")[-1] == "uroman"
        if is_uroman:
            logger.warning("For this checkpoint, you should use `uroman` to convert input text before tokenizing it!")
    else:
        logger.info(f"***Converting model: {checkpoint_path}***")
        is_uroman = False

    # original VITS checkpoint
    if vocab_path is None:
        _pad = "_"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        symbols = _pad + _punctuation + _letters + _letters_ipa
        {s: i for i, s in enumerate(symbols)}
    else:
        # Save vocab as temporary json file
        symbols = [line.replace("\n", "") for line in open(vocab_path, encoding="utf-8").readlines()]
        {s: i for i, s in enumerate(symbols)}
        # MMS-TTS does not use a <pad> token, so we set to the token used to space characters
        _pad = symbols[0]

    config.vocab_size = len(symbols)
    model = VitsDiscriminator(config)

    for disc in model.discriminators:
        disc.apply_weight_norm()

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # load weights

    state_dict = checkpoint["model"]

    for k, v in list(state_dict.items()):
        for old_layer_name in MAPPING:
            new_k = k.replace(old_layer_name, MAPPING[old_layer_name])

        state_dict[new_k] = state_dict.pop(k)

    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = {k for k in extra_keys if not k.endswith(".attn.bias")}
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = {k for k in missing_keys if not k.endswith(".attn.bias")}
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    model.load_state_dict(state_dict, strict=False)
    n_params = model.num_parameters(exclude_embeddings=True)
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    for disc in model.discriminators:
        disc.remove_weight_norm()

    model.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Local path to original checkpoint")
    parser.add_argument("--vocab_path", default=None, type=str, help="Path to vocab.txt")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument("--language", default=None, type=str, help="Tokenizer language (three-letter code)")
    parser.add_argument("--num_speakers", default=None, type=int, help="Number of speakers")
    parser.add_argument(
        "--sampling_rate", default=None, type=int, help="Sampling rate on which the model was trained."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.pytorch_dump_folder_path,
        args.checkpoint_path,
        args.config_path,
        args.vocab_path,
        args.language,
        args.num_speakers,
        args.sampling_rate,
        args.push_to_hub,
    )
