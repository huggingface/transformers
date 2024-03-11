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
from torch import nn

from transformers import MBartConfig, MBartForConditionalGeneration


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def convert_fairseq_mbart_checkpoint_from_disk(
    checkpoint_path, hf_config_path="facebook/mbart-large-en-ro", finetuned=False, mbart_50=False
):
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]

    mbart_config = MBartConfig.from_pretrained(hf_config_path, vocab_size=vocab_size)
    if mbart_50 and finetuned:
        mbart_config.activation_function = "relu"

    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    model = MBartForConditionalGeneration(mbart_config)
    model.model.load_state_dict(state_dict)

    if finetuned:
        model.lm_head = make_linear_from_emb(model.model.shared)

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
        help="Which huggingface architecture to use: mbart-large",
    )
    parser.add_argument("--mbart_50", action="store_true", help="whether the model is mMART-50 checkpoint")
    parser.add_argument("--finetuned", action="store_true", help="whether the model is a fine-tuned checkpoint")
    args = parser.parse_args()
    model = convert_fairseq_mbart_checkpoint_from_disk(
        args.fairseq_path, hf_config_path=args.hf_config, finetuned=args.finetuned, mbart_50=args.mbart_50
    )
    model.save_pretrained(args.pytorch_dump_folder_path)
