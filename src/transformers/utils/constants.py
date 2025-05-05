# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF2_WEIGHTS_INDEX_NAME = "tf_model.h5.index.json"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
PROCESSOR_NAME = "processor_config.json"
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"


SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

MULTIPLE_CHOICE_DUMMY_INPUTS = [
    [[0, 1, 0, 1], [1, 0, 0, 1]]
] * 2  # Needs to have 0s and 1s only since XLM uses it for langs too.
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]


__all__ = [
    "IMAGENET_DEFAULT_MEAN",
    "IMAGENET_DEFAULT_STD",
    "IMAGENET_STANDARD_MEAN",
    "IMAGENET_STANDARD_STD",
    "OPENAI_CLIP_MEAN",
    "OPENAI_CLIP_STD",
    "WEIGHTS_NAME",
    "WEIGHTS_INDEX_NAME",
    "TF2_WEIGHTS_NAME",
    "TF2_WEIGHTS_INDEX_NAME",
    "TF_WEIGHTS_NAME",
    "FLAX_WEIGHTS_NAME",
    "FLAX_WEIGHTS_INDEX_NAME",
    "SAFE_WEIGHTS_NAME",
    "SAFE_WEIGHTS_INDEX_NAME",
    "CONFIG_NAME",
    "FEATURE_EXTRACTOR_NAME",
    "IMAGE_PROCESSOR_NAME",
    "PROCESSOR_NAME",
    "GENERATION_CONFIG_NAME",
    "MODEL_CARD_NAME",
    "SENTENCEPIECE_UNDERLINE",
    "SPIECE_UNDERLINE",
    "MULTIPLE_CHOICE_DUMMY_INPUTS",
    "DUMMY_INPUTS",
    "DUMMY_MASK",
]
