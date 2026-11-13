# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

"""Lightweight constants for the Mistral native-format integration."""

import os


# File name of the native Mistral tekken tokenizer vocabulary. This remains the canonical
# name used for hub/directory discovery.
TEKKEN_VOCAB_FILE = "tekken.json"


def is_tekken_vocab_filename(path: str | os.PathLike) -> bool:
    """Return whether *path*'s filename identifies it as a Mistral tekken vocabulary file.

    Mirrors the classification rule of `mistral_common.tokens.tokenizers.tekken.is_tekken`:
    the basename must contain `"tekken"` and end with `.json`.

    Args:
        path (`str` or `os.PathLike`): Path (or bare filename) to classify.

    Returns:
        `bool`: `True` if the basename contains `"tekken"` and ends with `.json`.
    """
    basename = os.path.basename(path)
    return "tekken" in basename and basename.endswith(".json")
