# Copyright 2026 The ONDEWO Team. All rights reserved.
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
"""Compatibility shims that keep the retained Flax code working on transformers 5.x.

transformers 5.0.0 removed all Flax modeling code and, with it, a handful of constants and
helpers the Flax files still import. This fork keeps Flax Whisper (see
`models/whisper/modeling_flax_whisper.py`), so those few names are reimplemented here rather
than being patched back into upstream files - keeping the fork's diff against upstream to a
small set of Flax-only files that upstream no longer touches.

Everything in this module exists because upstream deleted it. Nothing here should grow into
general-purpose utility code.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional
from urllib.parse import urlparse


# --- constants removed alongside the Flax models -------------------------------------------
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"


# --- helpers removed in 5.x ----------------------------------------------------------------
def is_safetensors_available() -> bool:
    """safetensors is a hard dependency of transformers 5.x, so it is always importable."""
    return True


def is_flax_available() -> bool:
    """Whether jax and flax are importable.

    transformers 5.x dropped its own `is_flax_available` when the Flax models were removed,
    but this fork still gates its retained Flax exports on the optional dependency.
    """
    import importlib.util

    return importlib.util.find_spec("jax") is not None and importlib.util.find_spec("flax") is not None


def is_offline_mode() -> bool:
    """Mirror of the 4.x helper: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE force local-only loads."""
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        value: Optional[str] = os.environ.get(var)
        if value is not None and value.upper() in ("1", "ON", "YES", "TRUE"):
            return True
    return False


def is_remote_url(url_or_filename: Any) -> bool:
    return urlparse(str(url_or_filename)).scheme in ("http", "https")


def download_url(url: str, proxies: Optional[dict] = None) -> str:
    """Download `url` to a temp file and return the path (used only by the legacy load path)."""
    import requests

    tmp_fd, tmp_file = tempfile.mkstemp()
    with os.fdopen(tmp_fd, "wb") as f:
        response = requests.get(url, proxies=proxies, stream=True)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return tmp_file


# --- behavioural shims ---------------------------------------------------------------------
def patch_is_tensor_for_jax() -> None:
    """Teach `transformers.utils.generic.is_tensor` about jax arrays.

    transformers 5.x dropped jax from `is_tensor`, so it reports False for a `jnp.ndarray`.
    `ModelOutput.__post_init__` then treats the first field as a plain iterable and tries to
    `setattr` it back to None - which raises `FrozenInstanceError`, because every Flax model
    output is a frozen `@flax.struct.dataclass`. Widening `is_tensor` restores the 4.x path.

    Idempotent, and safe for the rest of transformers: jax arrays genuinely are tensors.
    """
    import jax

    from . import generic

    original = generic.is_tensor
    if getattr(original, "_jax_aware", False):
        return

    def is_tensor(x) -> bool:
        # Covers concrete arrays and jit tracers alike - both are instances of jax.Array.
        if isinstance(x, jax.Array):
            return True
        return original(x)

    is_tensor._jax_aware = True
    generic.is_tensor = is_tensor


# transformers 4.x GenerationConfig shipped these scalar defaults; 5.x sets them all to None.
# The Flax generation code does arithmetic and comparisons on them, so restore the old values
# for any field the loaded generation_config.json did not already set.
_FLAX_GENERATION_DEFAULTS = {
    "max_length": 20,
    "min_length": 0,
    "do_sample": False,
    "early_stopping": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "length_penalty": 1.0,
    "output_scores": False,
    "return_dict_in_generate": False,
}


def backfill_generation_config_defaults(generation_config) -> None:
    """Restore the transformers 4.x scalar defaults on a 5.x GenerationConfig, in place.

    `output_scores` matters here beyond upstream parity: this fork's Flax greedy and beam
    search read it to decide whether to accumulate per-token scores, and `None` would silently
    disable the confidence-score and alternatives feature.
    """
    for name, default in _FLAX_GENERATION_DEFAULTS.items():
        if getattr(generation_config, name, None) is None:
            setattr(generation_config, name, default)


__all__ = [
    "FLAX_WEIGHTS_INDEX_NAME",
    "FLAX_WEIGHTS_NAME",
    "backfill_generation_config_defaults",
    "download_url",
    "is_flax_available",
    "is_offline_mode",
    "is_remote_url",
    "is_safetensors_available",
    "patch_is_tensor_for_jax",
]
