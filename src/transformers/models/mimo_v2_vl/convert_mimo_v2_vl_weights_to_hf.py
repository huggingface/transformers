# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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
"""Convert original MiMo V2 series checkpoints to native, dequantized Transformers checkpoints."""

import argparse

from ...modeling_mimo_v2_utils import load_original_checkpoint
from ...tokenization_utils_tokenizers import TokenizersBackend
from .processing_mimo_v2_vl import MiMoV2Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = load_original_checkpoint(args.checkpoint, device_map=args.device)
    model.save_pretrained(args.output, max_shard_size="5GB")
    processor_class = MiMoV2Processor if model.config.model_type == "mimo_v2_vl" else TokenizersBackend
    processor_class.from_pretrained(args.checkpoint).save_pretrained(args.output)


if __name__ == "__main__":
    main()
