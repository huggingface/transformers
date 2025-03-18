# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

import numpy as np
import requests
from PIL import Image

from ..models.auto.auto_factory import _get_model_class
from ..models.auto.configuration_auto import AutoConfig
from ..models.auto.modeling_auto import MODEL_FOR_PRETRAINING_MAPPING, MODEL_MAPPING
from ..models.auto.processing_auto import PROCESSOR_MAPPING_NAMES, AutoProcessor
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES, AutoTokenizer
from .import_utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn

# Print the matrix with words as row labels
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BLACK_SQUARE = "■"
WHITE_SQUARE = "⬚"


def generate_sliding_window_mask_matrix(words, sliding_window=0, img_token="<img>", is_causal=True):
    n = len(words)
    max_word_length = max(len(word) for word in words)
    first_img_idx = 0

    if sliding_window != 0:
        mask = np.tril(np.zeros((n, n), dtype=int))
        for i in range(n):
            mask[i, max(0, i - sliding_window + 1) : i + 1] = 1
    else:
        if is_causal:
            mask = np.tril(np.ones((n, n), dtype=int))
        else:
            mask = np.ones((n, n), dtype=int)

    for i, k in enumerate(words):
        if img_token in k and not first_img_idx:
            first_img_idx = i
        if first_img_idx > 0 and (img_token not in k or i == n - 1):
            if i == n - 1:
                i += 1
            mask[first_img_idx:i, first_img_idx:i] = 1
            first_img_idx = 0

    vertical_header = []
    for idx, word in enumerate(words):
        if img_token not in word:
            vertical_header += [list(str(idx).rjust(len(str(n))))]
        else:
            vertical_header += [[f"{YELLOW}{k}{RESET}" for k in list(str(idx).rjust(len(str(n))))]]

    vertical_header = list(map(list, zip(*vertical_header)))  # Transpose
    # Print the vertical header
    for row in vertical_header:
        print((max_word_length + 5) * " " + " ".join(row))

    for i, word in enumerate(words):
        colored_word = (
            f"{YELLOW}{word.ljust(max_word_length)}{RESET}" if img_token in word else word.ljust(max_word_length)
        )
        number = str(i).rjust(len(str(n)))
        colored_number = f"{YELLOW}{number}{RESET}" if img_token in word else number
        base_display = colored_word + ": " + colored_number + " "
        row_display = " ".join(
            f"{YELLOW}{BLACK_SQUARE}{RESET}"
            if img_token in words[j] and mask[i, j] and img_token in words[i]
            else f"{GREEN}{BLACK_SQUARE}{RESET}"
            if i == j
            else BLACK_SQUARE
            if mask[i, j]
            else WHITE_SQUARE
            for j in range(n)
        )
        print(base_display + row_display)

    print(" " * len(base_display) + "-" * len(row_display))


# sentece = (
#     "What is this? <img> <img> <img> <img> This is a cat. And these ? <img> <img> <img> <img> <img> These are dogs."
# )
# words = sentece.split()
# generate_sliding_window_mask_matrix(words)
# generate_sliding_window_mask_matrix(words, sliding_window=3)


# sentece = (
#     "<img> <img> <img> <img> This is the system prompt."
# )
# words = sentece.split()
# generate_sliding_window_mask_matrix(words, is_causal=False)

# sentece = (
#     "This step is decoding, the mask is causal."
# )
# words = sentece.split()
# generate_sliding_window_mask_matrix(words, is_causal=True)

# sentece = (
#     "<img> <img> <img> <img> This is the system prompt. This step is decoding, the mask is causal."
# )
# words = sentece.split()
# generate_sliding_window_mask_matrix(words, is_causal=True)


# Should print:
""""


                              1 1 1 1 1 1 1 1 1 1 2 2
          0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
What :  0 ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
is   :  1 ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
this?:  2 ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  3 ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  4 ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  5 ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  6 ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
This :  7 ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
is   :  8 ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
a    :  9 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
cat. : 10 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
And  : 11 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
these: 12 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
?    : 13 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>: 14 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 15 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 16 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 17 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 18 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
These: 19 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚
are  : 20 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚
dogs.: 21 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
          ----------------------------------------------------
                              1 1 1 1 1 1 1 1 1 1 2 2
          0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
What :  0 ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
is   :  1 ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
this?:  2 ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  3 ⬚ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  4 ⬚ ⬚ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  5 ⬚ ⬚ ⬚ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  6 ⬚ ⬚ ⬚ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
This :  7 ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
is   :  8 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
a    :  9 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
cat. : 10 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
And  : 11 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
these: 12 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
?    : 13 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>: 14 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 15 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 16 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 17 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 18 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
These: 19 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚
are  : 20 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚
dogs.: 21 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■
          ----------------------------------------------------

"""


def generate_attention_matrix_from_mask(words, mask, img_token="<img>"):
    mask = mask.int()
    if mask.ndim == 3:
        mask = mask[0, :, :]
    if mask.ndim == 4:
        mask = mask[0, 0, :, :]

    n = len(words)
    max_word_length = max(len(word) for word in words)
    first_img_idx = 0

    for i, k in enumerate(words):
        if img_token in k and not first_img_idx:
            first_img_idx = i
        if first_img_idx > 0 and (img_token not in k or i == n - 1):
            if i == n - 1:
                i += 1
            mask[first_img_idx:i, first_img_idx:i] = 2
            first_img_idx = 0

    vertical_header = []
    for idx, word in enumerate(words):
        if img_token not in word:
            vertical_header += [list(str(idx).rjust(len(str(n))))]
        else:
            vertical_header += [[f"{YELLOW}{k}{RESET}" for k in list(str(idx).rjust(len(str(n))))]]

    vertical_header = list(map(list, zip(*vertical_header)))  # Transpose
    # Print the vertical header
    for row in vertical_header:
        print((max_word_length + 5) * " " + " ".join(row))

    for i, word in enumerate(words):
        colored_word = (
            f"{YELLOW}{word.ljust(max_word_length)}{RESET}" if img_token in word else word.ljust(max_word_length)
        )
        number = str(i).rjust(len(str(n)))
        colored_number = f"{YELLOW}{number}{RESET}" if img_token in word else number
        base_display = colored_word + ": " + colored_number + " "
        row_display = " ".join(
            f"{YELLOW}{BLACK_SQUARE}{RESET}"
            if img_token in words[j] and mask[i, j] and img_token in words[i]
            else f"{GREEN}{BLACK_SQUARE}{RESET}"
            if i == j
            else BLACK_SQUARE
            if mask[i, j]
            else WHITE_SQUARE
            for j in range(n)
        )
        print(base_display + row_display)

    print(" " * len(base_display) + "-" * len(row_display))


class AttentionMaskVisualizer:
    def __init__(self, model_name: str):
        config = AutoConfig.from_pretrained(model_name)
        self.config = config
        if hasattr(config, "sliding_window"):
            config.sliding_window = 5

        try:
            mapped_cls = _get_model_class(config, MODEL_MAPPING)
        except Exception:
            mapped_cls = _get_model_class(config, MODEL_FOR_PRETRAINING_MAPPING)

        if mapped_cls is None:
            raise ValueError(f"Model name {model_name} is not supported for attention visualization")

        class _ModelWrapper(mapped_cls, nn.Module):
            def __init__(self, config, model_name):
                nn.Module.__init__(self)
                self.dummy_module = nn.Linear(1, 1)
                self.config = config

        self.model = _ModelWrapper(config, model_name)
        self.model.to(config.torch_dtype)
        self.repo_id = model_name

    def __call__(self, input_sentence: str):
        self.visualize_attention_mask(input_sentence)

    def visualize_attention_mask(self, input_sentence: str):
        model = self.model
        kwargs = {}

        if self.config.model_type in PROCESSOR_MAPPING_NAMES:
            img = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"
            img = Image.open(requests.get(img, stream=True).raw)
            kwargs["text"] = input_sentence
            processor = AutoProcessor.from_pretrained(self.repo_id)
            image_token = processor.tokenizer.convert_ids_to_tokens([processor.image_token_id])[0]

            if image_token:
                input_sentence = input_sentence.replace("<img>", image_token)

            attention_mask = processor(img, input_sentence, return_tensors="pt")["attention_mask"]
            tokens = processor.tokenizer.tokenize(input_sentence)
        elif self.config.model_type in TOKENIZER_MAPPING_NAMES:
            tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
            tokens = tokenizer.tokenize(input_sentence)
            attention_mask = tokenizer(input_sentence, return_tensors="pt")["attention_mask"]
        else:
            raise ValueError(f"Model type {model.config.model_type} does not support attention visualization")

        model.config._attn_implementation = "eager"

        attention_mask = ~model._update_causal_mask(
            attention_mask=attention_mask,
            input_tensor=attention_mask.to(self.model.dtype),
            cache_position=torch.arange(attention_mask.shape[1]),
            past_key_values=None,
        ).bool()

        generate_attention_matrix_from_mask(tokens, attention_mask)
