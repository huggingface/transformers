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
"""Prepare the original Xiaomi-Lens-0 checkpoint for native Transformers loading.

Only metadata is changed. Weight files, the original architecture name, Qwen sub-configs,
and the legacy mask_decoder dictionary are preserved for the original implementation.
The native model converts decoder weights on load using conversion_mapping.py.

The original repo must explicitly load Qwen3VLProcessor when using an older Transformers
release that does not provide XiaomiLens0Processor. Existing remote-code files and auto_map
entries are copied, but this script does not create a remote implementation.

Usage:
    python src/transformers/models/xiaomi_lens0/convert_xiaomi_lens0_weights_to_hf.py \
        --input_path SeerRay-Lab/Xiaomi-Lens-0 --output_path Xiaomi-Lens-0-hf
"""

import argparse
import copy
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from transformers import AutoTokenizer, XiaomiLens0Config


def convert_config(original_config: dict, mask_token_id: int) -> dict:
    config = copy.deepcopy(original_config)
    config["model_type"] = "xiaomi_lens0"
    config["mask_token_id"] = mask_token_id
    # Check that the native config accepts the legacy schema without serializing it back:
    # serialization would replace the Qwen sub-config types and legacy decoder field names.
    XiaomiLens0Config.from_dict(config)
    return config


def convert_checkpoint(input_path: str, output_path: str, push_to_hub: str | None = None, revision: str | None = None):
    output_dir = Path(output_path).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output_path must be an empty directory or a new path.")
    source_dir = Path(input_path)
    if not source_dir.is_dir():
        source_dir = Path(snapshot_download(input_path, revision=revision))
    source_dir = source_dir.resolve()
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise ValueError("output_path must not be inside the input checkpoint.")

    original_config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    config = convert_config(original_config, mask_token_id=tokenizer.convert_tokens_to_ids("<|mask|>"))
    config["think_start_token_id"] = tokenizer.convert_tokens_to_ids("<think>")
    config["think_end_token_id"] = tokenizer.convert_tokens_to_ids("</think>")

    # Copy bytes instead of loading and re-saving tensors, preserving shard names, dtypes,
    # indexes, and the original fused attention projections exactly.
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".cache"))
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for filename in (
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer_config.json",
    ):
        path = output_dir / filename
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if filename == "processor_config.json" or "processor_class" in metadata:
            metadata["processor_class"] = "XiaomiLens0Processor"
            path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if push_to_hub is not None:
        api = HfApi()
        api.create_repo(push_to_hub, private=True, exist_ok=True)
        api.upload_folder(repo_id=push_to_hub, folder_path=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="SeerRay-Lab/Xiaomi-Lens-0", help="Original checkpoint repo or path.")
    parser.add_argument("--output_path", required=True, help="New or empty directory for the converted checkpoint.")
    parser.add_argument("--revision", default=None, help="Source Hub revision to download.")
    parser.add_argument("--push_to_hub", default=None, help="Optional repo id to push the converted checkpoint to.")
    args = parser.parse_args()
    convert_checkpoint(args.input_path, args.output_path, args.push_to_hub, args.revision)
