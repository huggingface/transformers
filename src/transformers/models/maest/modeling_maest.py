# coding=utf-8
# Copyright 2023 MIT and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch MAEST (MAEST) model."""



from ...utils import logging


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "MAESTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "mtg-upf/discogs-maest-30s-pw-129e"
_EXPECTED_OUTPUT_SHAPE = [1, 1214, 768]

# Audio classification docstring
_SEQ_CLASS_CHECKPOINT = "mtg-upf/discogs-maest-30s-pw-129e"
_SEQ_CLASS_EXPECTED_OUTPUT = "'Music'"
_SEQ_CLASS_EXPECTED_LOSS = 0.00


MAEST_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "mtg-upf/discogs-maest-30s-pw-129e",
    "mtg-upf/discogs-maest-10s-fs-129e",
    "mtg-upf/discogs-maest-10s-pw-129e",
    "mtg-upf/discogs-maest-10s-dw-75e",
    "mtg-upf/discogs-maest-5s-pw-129e",
    "mtg-upf/discogs-maest-20s-pw-129e",
    "mtg-upf/discogs-maest-30s-pw-129e",
    "mtg-upf/discogs-maest-30s-pw-73e-ts",
]
