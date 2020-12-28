# Copyright 2020-present, the HuggingFace Inc. team.
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
Once a model has been fine-pruned, the weights that are masked during the forward pass can be pruned once for all.
For instance, once the a model from the :class:`~emmental.MaskedBertForSequenceClassification` is trained, it can be saved (and then loaded)
as a standard :class:`~transformers.BertForSequenceClassification`.
"""

import argparse
import os
import shutil

import torch

from emmental.modules import MagnitudeBinarizer, ThresholdBinarizer, TopKBinarizer


def main(args):
    pruning_method = args.pruning_method
    threshold = args.threshold

    model_name_or_path = args.model_name_or_path.rstrip("/")
    target_model_path = args.target_model_path

    print(f"Load fine-pruned model from {model_name_or_path}")
    model = torch.load(os.path.join(model_name_or_path, "pytorch_model.bin"))
    pruned_model = {}

    for name, tensor in model.items():
        if "embeddings" in name or "LayerNorm" in name or "pooler" in name:
            pruned_model[name] = tensor
            print(f"Copied layer {name}")
        elif "classifier" in name or "qa_output" in name:
            pruned_model[name] = tensor
            print(f"Copied layer {name}")
        elif "bias" in name:
            pruned_model[name] = tensor
            print(f"Copied layer {name}")
        else:
            if pruning_method == "magnitude":
                mask = MagnitudeBinarizer.apply(inputs=tensor, threshold=threshold)
                pruned_model[name] = tensor * mask
                print(f"Pruned layer {name}")
            elif pruning_method == "topK":
                if "mask_scores" in name:
                    continue
                prefix_ = name[:-6]
                scores = model[f"{prefix_}mask_scores"]
                mask = TopKBinarizer.apply(scores, threshold)
                pruned_model[name] = tensor * mask
                print(f"Pruned layer {name}")
            elif pruning_method == "sigmoied_threshold":
                if "mask_scores" in name:
                    continue
                prefix_ = name[:-6]
                scores = model[f"{prefix_}mask_scores"]
                mask = ThresholdBinarizer.apply(scores, threshold, True)
                pruned_model[name] = tensor * mask
                print(f"Pruned layer {name}")
            elif pruning_method == "l0":
                if "mask_scores" in name:
                    continue
                prefix_ = name[:-6]
                scores = model[f"{prefix_}mask_scores"]
                l, r = -0.1, 1.1
                s = torch.sigmoid(scores)
                s_bar = s * (r - l) + l
                mask = s_bar.clamp(min=0.0, max=1.0)
                pruned_model[name] = tensor * mask
                print(f"Pruned layer {name}")
            else:
                raise ValueError("Unknown pruning method")

    if target_model_path is None:
        target_model_path = os.path.join(
            os.path.dirname(model_name_or_path), f"bertarized_{os.path.basename(model_name_or_path)}"
        )

    if not os.path.isdir(target_model_path):
        shutil.copytree(model_name_or_path, target_model_path)
        print(f"\nCreated folder {target_model_path}")

    torch.save(pruned_model, os.path.join(target_model_path, "pytorch_model.bin"))
    print("\nPruned model saved! See you later!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pruning_method",
        choices=["l0", "magnitude", "topK", "sigmoied_threshold"],
        type=str,
        required=True,
        help="Pruning Method (l0 = L0 regularization, magnitude = Magnitude pruning, topK = Movement pruning, sigmoied_threshold = Soft movement pruning)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        help="For `magnitude` and `topK`, it is the level of remaining weights (in %) in the fine-pruned model."
        "For `sigmoied_threshold`, it is the threshold \tau against which the (sigmoied) scores are compared."
        "Not needed for `l0`",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Folder containing the model that was previously fine-pruned",
    )
    parser.add_argument(
        "--target_model_path",
        default=None,
        type=str,
        required=False,
        help="Folder containing the model that was previously fine-pruned",
    )

    args = parser.parse_args()

    main(args)
