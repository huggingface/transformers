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


import requests
from PIL import Image

from ..masking_utils import create_causal_mask
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


def generate_attention_matrix_from_mask(
    words, mask, img_token="<img>", sliding_window=None, token_type_ids=None, image_seq_length=None
):
    """
    Generates an attention matrix from a given attention mask.

    Optionally applies a sliding window mask (e.g., for Gemma2/3) and
    marks regions where image tokens occur based on the specified `img_token`.
    """
    mask = mask.int()
    if mask.ndim == 3:
        mask = mask[0, :, :]
    if mask.ndim == 4:
        mask = mask[0, 0, :, :]

    n = len(words)
    max_word_length = max(len(repr(word)) for word in words)
    first_img_idx = 0
    output = []

    for i, k in enumerate(words):
        if k == img_token and not first_img_idx:
            first_img_idx = i
            mask[i, i] = 2  # Mark yellow regions
        if first_img_idx > 0 and (k != img_token or i == n - 1):
            if i == n - 1:
                i += 1
            mask[first_img_idx:i, first_img_idx:i] = 2  # Mark yellow regions
            first_img_idx = 0

    # Generate sliding window mask (size = 4), excluding img_token
    sliding_window_mask = None
    if sliding_window is not None:
        sliding_window_mask = [[1 if (0 <= i - j < sliding_window) else 0 for j in range(n)] for i in range(n)]

    row_dummy = " ".join(
        f"{YELLOW}{BLACK_SQUARE}{RESET}"
        if mask[0, j]
        else f"{GREEN}{BLACK_SQUARE}{RESET}"
        if 0 == j
        else BLACK_SQUARE
        if mask[0, j]
        else WHITE_SQUARE
        for j in range(n)
    )

    if token_type_ids is not None:
        is_special = token_type_ids == 1
        token_type_buckets = torch.where(
            (token_type_ids.cumsum(-1) % 5 + is_special).bool(), token_type_ids.cumsum(-1), 0
        )
        boundaries = torch.arange(0, image_seq_length + 1, image_seq_length)
        token_type_buckets = torch.bucketize(token_type_buckets, boundaries=boundaries)

    # Print headers
    legend = f"{GREEN}{BLACK_SQUARE}{RESET}: i == j (diagonal)   {YELLOW}{BLACK_SQUARE}{RESET}: token_type_ids"
    output.append(" " + legend)
    f_string = " " * (max_word_length + 5) + "Attention Matrix".ljust(len(row_dummy) // 2)
    if sliding_window is not None:
        f_string += "Sliding Window Mask"
    output.append(f_string)

    vertical_header = []
    for idx, word in enumerate(words):
        if mask[idx, idx] == 2:
            vertical_header.append([f"{YELLOW}{k}{RESET}" for k in list(str(idx).rjust(len(str(n))))])
        else:
            vertical_header.append(list(str(idx).rjust(len(str(n)))))

    vertical_header = list(map(list, zip(*vertical_header)))  # Transpose

    for row in vertical_header:
        output.append(
            (max_word_length + 5) * " " + " ".join(row) + "    |    " + " ".join(row)
            if sliding_window is not None
            else ""
        )
    for i, word in enumerate(words):
        word_repr = repr(word).ljust(max_word_length)
        colored_word = f"{YELLOW}{word_repr}{RESET}" if img_token in word else word_repr
        row_display = " ".join(
            f"{YELLOW}{BLACK_SQUARE}{RESET}"
            if img_token in words[j] and mask[i, j] and img_token in word
            else f"{GREEN}{BLACK_SQUARE}{RESET}"
            if i == j
            else BLACK_SQUARE
            if mask[i, j]
            else WHITE_SQUARE
            for j in range(n)
        )
        sliding_window_row = ""
        if sliding_window is not None:
            sliding_window_row = " ".join(
                f"{YELLOW}{BLACK_SQUARE}{RESET}"
                if img_token in words[j] and img_token in word and token_type_buckets[0, i] == token_type_buckets[0, j]
                else f"{GREEN}{BLACK_SQUARE}{RESET}"
                if i == j
                else BLACK_SQUARE
                if sliding_window_mask[i][j]
                else WHITE_SQUARE
                for j in range(n)
            )

        output.append(f"{colored_word}: {str(i).rjust(2)} {row_display}    |    {sliding_window_row}")

    return "\n".join(output)


class AttentionMaskVisualizer:
    def __init__(self, model_name: str):
        config = AutoConfig.from_pretrained(model_name)
        self.image_token = "<img>"
        if hasattr(config.get_text_config(), "sliding_window"):
            self.sliding_window = getattr(config.get_text_config(), "sliding_window", None)
        try:
            mapped_cls = _get_model_class(config, MODEL_MAPPING)
        except Exception:
            mapped_cls = _get_model_class(config, MODEL_FOR_PRETRAINING_MAPPING)

        if mapped_cls is None:
            raise ValueError(f"Model name {model_name} is not supported for attention visualization")
        self.mapped_cls = mapped_cls

        class _ModelWrapper(mapped_cls, nn.Module):
            def __init__(self, config, model_name):
                nn.Module.__init__(self)
                self.dummy_module = nn.Linear(1, 1)
                self.config = config

        self.model = _ModelWrapper(config, model_name)
        self.model.to(config.dtype)
        self.repo_id = model_name
        self.config = config

    def __call__(self, input_sentence: str, suffix=""):
        self.visualize_attention_mask(input_sentence, suffix=suffix)

    def visualize_attention_mask(self, input_sentence: str, suffix=""):
        model = self.model
        kwargs = {}
        image_seq_length = None
        if self.config.model_type in PROCESSOR_MAPPING_NAMES:
            img = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"
            img = Image.open(requests.get(img, stream=True).raw)
            image_seq_length = 5
            processor = AutoProcessor.from_pretrained(self.repo_id, image_seq_length=image_seq_length)
            if hasattr(processor, "image_token"):
                image_token = processor.image_token
            else:
                image_token = processor.tokenizer.convert_ids_to_tokens([processor.image_token_id])[0]

            if image_token:
                input_sentence = input_sentence.replace("<img>", image_token)

            inputs = processor(images=img, text=input_sentence, suffix=suffix, return_tensors="pt")

            self.image_token = processor.tokenizer.convert_ids_to_tokens([processor.image_token_id])[0]

            attention_mask = inputs["attention_mask"]
            if "token_type_ids" in inputs:  # TODO inspect signature of update causal mask
                kwargs["token_type_ids"] = inputs["token_type_ids"]
            tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        elif self.config.model_type in TOKENIZER_MAPPING_NAMES:
            tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
            tokens = tokenizer.tokenize(input_sentence)
            attention_mask = tokenizer(input_sentence, return_tensors="pt")["attention_mask"]
        else:
            raise ValueError(f"Model type {model.config.model_type} does not support attention visualization")

        model.config._attn_implementation = "eager"
        model.train()

        batch_size, seq_length = attention_mask.shape
        input_embeds = torch.zeros((batch_size, seq_length, model.config.hidden_size), dtype=self.model.dtype)
        cache_position = torch.arange(seq_length)

        causal_mask = create_causal_mask(
            config=model.config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
        )

        if causal_mask is not None:
            attention_mask = ~causal_mask.bool()
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, seq_length, seq_length)
        top_bottom_border = "##" * (
            len(f"Attention visualization for {self.config.model_type} | {self.mapped_cls}") + 4
        )  # Box width adjusted to text length
        side_border = "##"
        print(f"\n{top_bottom_border}")
        print(
            "##"
            + f"  Attention visualization for \033[1m{self.config.model_type}:{self.repo_id}\033[0m {self.mapped_cls.__name__}".center(
                len(top_bottom_border)
            )
            + "    "
            + side_border,
        )
        print(f"{top_bottom_border}")
        f_string = generate_attention_matrix_from_mask(
            tokens,
            attention_mask,
            img_token=self.image_token,
            sliding_window=getattr(self.config, "sliding_window", None),
            token_type_ids=kwargs.get("token_type_ids"),
            image_seq_length=image_seq_length,
        )
        print(f_string)
        print(f"{top_bottom_border}")
