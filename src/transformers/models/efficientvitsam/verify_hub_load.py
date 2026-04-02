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
"""
Smoke-test a Hub repo after upload: load `EfficientvitsamModel` + `EfficientvitsamProcessor` and run one forward.

Usage (after `huggingface-cli login` if the repo is private):

    python -m transformers.models.efficientvitsam.verify_hub_load YOUR_ORG/your-model-name
"""

import argparse
import sys

import numpy as np
import torch

from transformers import AutoModel, AutoProcessor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", type=str, help="Hub repo id, e.g. org/model-name")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the forward pass",
    )
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.repo_id)
    model = AutoModel.from_pretrained(args.repo_id).to(args.device)
    model.eval()

    # Random RGB image (processor handles resize/normalize like SAM)
    images = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    inputs = processor(images=images, return_tensors="pt").to(args.device)

    with torch.no_grad():
        out = model(**inputs)

    assert out.iou_scores is not None
    print(f"OK: loaded {args.repo_id}, iou_scores shape = {tuple(out.iou_scores.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
