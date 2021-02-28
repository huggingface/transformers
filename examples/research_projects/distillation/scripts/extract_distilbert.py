# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
"""
Preprocessing script before training DistilBERT.
Specific to BERT -> DistilBERT.
"""
import argparse

import torch

from transformers import BertForMaskedLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction some layers of the full BertForMaskedLM or RObertaForMaskedLM for Transfer Learned Distillation"
    )
    parser.add_argument("--model_type", default="bert", choices=["bert"])
    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument("--dump_checkpoint", default="serialization_dir/tf_bert-base-uncased_0247911.pth", type=str)
    parser.add_argument("--vocab_transform", action="store_true")
    args = parser.parse_args()

    if args.model_type == "bert":
        model = BertForMaskedLM.from_pretrained(args.model_name)
        prefix = "bert"
    else:
        raise ValueError('args.model_type should be "bert".')

    state_dict = model.state_dict()
    compressed_sd = {}

    for w in ["word_embeddings", "position_embeddings"]:
        compressed_sd[f"distilbert.embeddings.{w}.weight"] = state_dict[f"{prefix}.embeddings.{w}.weight"]
    for w in ["weight", "bias"]:
        compressed_sd[f"distilbert.embeddings.LayerNorm.{w}"] = state_dict[f"{prefix}.embeddings.LayerNorm.{w}"]

    std_idx = 0
    for teacher_idx in [0, 2, 4, 7, 9, 11]:
        for w in ["weight", "bias"]:
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.attention.q_lin.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.query.{w}"
            ]
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.attention.k_lin.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.key.{w}"
            ]
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.attention.v_lin.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.value.{w}"
            ]

            compressed_sd[f"distilbert.transformer.layer.{std_idx}.attention.out_lin.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.output.dense.{w}"
            ]
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.sa_layer_norm.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.output.LayerNorm.{w}"
            ]

            compressed_sd[f"distilbert.transformer.layer.{std_idx}.ffn.lin1.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.intermediate.dense.{w}"
            ]
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.ffn.lin2.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.output.dense.{w}"
            ]
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.output_layer_norm.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.output.LayerNorm.{w}"
            ]
        std_idx += 1

    compressed_sd["vocab_projector.weight"] = state_dict["cls.predictions.decoder.weight"]
    compressed_sd["vocab_projector.bias"] = state_dict["cls.predictions.bias"]
    if args.vocab_transform:
        for w in ["weight", "bias"]:
            compressed_sd[f"vocab_transform.{w}"] = state_dict[f"cls.predictions.transform.dense.{w}"]
            compressed_sd[f"vocab_layer_norm.{w}"] = state_dict[f"cls.predictions.transform.LayerNorm.{w}"]

    print(f"N layers selected for distillation: {std_idx}")
    print(f"Number of params transferred for distillation: {len(compressed_sd.keys())}")

    print(f"Save transferred checkpoint to {args.dump_checkpoint}.")
    torch.save(compressed_sd, args.dump_checkpoint)
