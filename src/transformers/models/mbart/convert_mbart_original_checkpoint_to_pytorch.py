# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import torch

from transformers import BartForConditionalGeneration, MBartConfig

from ..bart.convert_bart_original_pytorch_checkpoint_to_pytorch import remove_ignore_keys_


def convert_fairseq_mbart_checkpoint_from_disk(checkpoint_path, hf_config_path="facebook/mbart-large-en-ro"):
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]
    mbart_config = MBartConfig.from_pretrained(hf_config_path, vocab_size=vocab_size)
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    model = BartForConditionalGeneration(mbart_config)
    model.model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config",
        default="facebook/mbart-large-cc25",
        type=str,
        help="Which huggingface architecture to use: bart-large-xsum",
    )
    args = parser.parse_args()
    model = convert_fairseq_mbart_checkpoint_from_disk(args.fairseq_path, hf_config_path=args.hf_config)
    model.save_pretrained(args.pytorch_dump_folder_path)
