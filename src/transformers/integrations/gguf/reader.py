# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Reading a GGUF file: architecture, tensor types, and tensors keyed by their GGUF names.

Renaming and any conversion are *not* done here — they are `WeightConverter`s in `archs.py`, run by
the normal loading pipeline. This module only turns a file into `{gguf_name: tensor}`.
"""

import re

import numpy as np
import torch

from ...utils import is_gguf_available, logging


logger = logging.get_logger(__name__)

# ggml type ids for the types a non-quantized GGUF can hold
_GGML_F32, _GGML_F16, _GGML_BF16 = 0, 1, 30

_TORCH_DTYPE = {_GGML_F32: torch.float32, _GGML_F16: torch.float16, _GGML_BF16: torch.bfloat16}


def read_gguf_architecture(gguf_path: str) -> str:
    """Read only the `general.architecture` field, without touching tensor data."""
    reader = _reader(gguf_path)
    return str(reader.fields["general.architecture"].contents())


def read_gguf_tensor_types(gguf_path: str) -> dict[str, int]:
    """`{gguf_name: ggml_type}` from the header, without reading tensor data."""
    return {tensor.name: int(tensor.tensor_type) for tensor in _reader(gguf_path).tensors}


def unused_gguf_tensors(tensor_names, config) -> dict[str, str]:
    """`{gguf_name: reason}` for tensors this model has no parameter for.

    A GGUF file can hold blocks beyond the model's decoder stack — Qwen3.5 ships a
    multi-token-prediction block, for instance. Those are legitimately unused, but they must be
    *declared* rather than silently dropped, so that a genuinely missing renaming (which looks
    identical from the outside) is still an error.
    """
    n_layers = config.get_text_config().num_hidden_layers
    unused = {}
    for name in tensor_names:
        match = re.match(r"^blk\.(\d+)\.", name)
        if match is not None and int(match.group(1)) >= n_layers:
            unused[name] = (
                f"block {match.group(1)} is beyond the model's {n_layers} layers "
                "(e.g. a multi-token-prediction head)"
            )
    return unused


def load_gguf_state_dict(gguf_path: str, config=None) -> dict[str, torch.Tensor]:
    """`{gguf_name: tensor}` for a non-quantized GGUF file.

    ggml stores shapes reversed relative to torch, so they are flipped back here — that is a
    property of the container, not of any architecture, which is why it does not belong in a
    converter.

    When `config` is given, tensors this model has no place for are dropped and reported (see
    `unused_gguf_tensors`), instead of travelling through the pipeline to be silently discarded.
    """
    state_dict = {}
    for tensor in _reader(gguf_path).tensors:
        ggml_type = int(tensor.tensor_type)
        if ggml_type not in _TORCH_DTYPE:
            raise ValueError(
                f"{tensor.name} has ggml type {ggml_type}, which is quantized. Loading quantized GGUF "
                "files is not supported yet; use a bf16/f16/f32 GGUF."
            )
        # GGUFReader hands back read-only mmap views: copy before wrapping, or a later
        # `.to(device, dtype)` may treat that storage as scratch.
        data = torch.from_numpy(np.copy(tensor.data))
        shape = tuple(int(dim) for dim in reversed(tensor.shape))
        state_dict[tensor.name] = data.view(_TORCH_DTYPE[ggml_type]).reshape(shape)

    if config is not None:
        unused = unused_gguf_tensors(state_dict, config)
        for name in unused:
            del state_dict[name]
        if unused:
            reasons = sorted(set(unused.values()))
            logger.info(
                f"Skipped {len(unused)} tensor(s) from the GGUF file that this model has no parameter "
                f"for: {'; '.join(reasons)}."
            )
    return state_dict


def _reader(gguf_path: str):
    if not is_gguf_available():
        raise ImportError("Loading a GGUF file requires the `gguf` package. Run `pip install gguf`.")
    from gguf import GGUFReader

    return GGUFReader(gguf_path)
