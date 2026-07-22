# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""Convert original MOSS-TTS-Delay configs to the Transformers format."""

import argparse
import json
import os
from typing import Any

from transformers import MossTTSDelayConfig
from transformers.utils import logging
from transformers.utils.hub import cached_file


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


CONFIG_KEY_MAPPING = {
    "n_vq": "n_codebooks",
    "audio_vocab_size": "codebook_size",
    "audio_pad_code": "codebook_pad_token_id",
}


def _rename_config_key(config_dict: dict[str, Any], old_key: str, new_key: str) -> None:
    if old_key not in config_dict:
        return
    if new_key in config_dict and config_dict[new_key] != config_dict[old_key]:
        raise ValueError(
            f"Received both `{old_key}` and `{new_key}` with different values. Please keep only `{new_key}`."
        )
    config_dict[new_key] = config_dict.pop(old_key)


def convert_config(config_dict: dict[str, Any]) -> MossTTSDelayConfig:
    config_dict = dict(config_dict)
    for old_key, new_key in CONFIG_KEY_MAPPING.items():
        _rename_config_key(config_dict, old_key, new_key)

    return MossTTSDelayConfig(**config_dict)


def _load_config(input_path_or_repo: str, revision: str | None = None) -> dict[str, Any]:
    config_path = (
        input_path_or_repo
        if os.path.isfile(input_path_or_repo)
        else cached_file(input_path_or_repo, "config.json", revision=revision)
    )
    with open(config_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path_or_repo",
        type=str,
        required=True,
        help="Path to an original config.json file or a Hub repo id containing one.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the converted config is saved.")
    parser.add_argument("--revision", type=str, default=None, help="Hub revision to use when loading from a repo id.")
    args = parser.parse_args()

    config = convert_config(_load_config(args.input_path_or_repo, revision=args.revision))
    os.makedirs(args.output_dir, exist_ok=True)
    config.save_pretrained(args.output_dir)
    logger.info("Saved converted MOSS-TTS-Delay config to %s", args.output_dir)


if __name__ == "__main__":
    main()
