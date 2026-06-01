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

import numpy as np

from transformers.modeling_gguf_pytorch_utils import _dequantize_gguf_tensor, _dequantize_prism_q1_0_g128


class FakeQuantType:
    def __init__(self, name: str, value: int):
        self.name = name
        self._value = value

    def __int__(self):
        return self._value


def _pack_prism_block(scale: float, signs: np.ndarray) -> np.ndarray:
    sign_bytes = np.packbits(np.asarray(signs, dtype=np.uint8), bitorder="little")
    return np.concatenate([np.asarray([scale], dtype=np.float16).view(np.uint8), sign_bytes])


def _build_prism_rows():
    row0_block0_signs = (np.arange(128) % 2).astype(np.uint8)
    row0_block1_signs = (np.arange(128) % 3 == 0).astype(np.uint8)
    row1_block0_signs = np.ones(128, dtype=np.uint8)
    row1_block1_signs = np.zeros(128, dtype=np.uint8)

    row0 = np.concatenate(
        [
            _pack_prism_block(1.5, row0_block0_signs),
            _pack_prism_block(0.25, row0_block1_signs),
        ]
    )
    row1 = np.concatenate(
        [
            _pack_prism_block(2.0, row1_block0_signs),
            _pack_prism_block(0.75, row1_block1_signs),
        ]
    )
    data = np.stack([row0, row1], axis=0)
    expected = np.stack(
        [
            np.concatenate(
                [
                    np.where(row0_block0_signs == 1, np.float32(1.5), np.float32(-1.5)),
                    np.where(row0_block1_signs == 1, np.float32(0.25), np.float32(-0.25)),
                ]
            ),
            np.concatenate(
                [
                    np.where(row1_block0_signs == 1, np.float32(2.0), np.float32(-2.0)),
                    np.where(row1_block1_signs == 1, np.float32(0.75), np.float32(-0.75)),
                ]
            ),
        ],
        axis=0,
    )
    return data, expected


def test_dequantize_prism_q1_0_g128_matches_reference_layout():
    data, expected = _build_prism_rows()
    actual = _dequantize_prism_q1_0_g128(data)
    np.testing.assert_array_equal(actual, expected)


def test_dequantize_gguf_tensor_falls_back_for_prism_q1_0_g128():
    data, expected = _build_prism_rows()

    def fake_dequantize(_data, _tensor_type):
        raise NotImplementedError("missing q1_0_g128 support")

    actual = _dequantize_gguf_tensor(data, FakeQuantType("Q1_0_g128", 41), fake_dequantize)
    np.testing.assert_array_equal(actual, expected)


def test_dequantize_gguf_tensor_uses_default_path_for_other_quant_types():
    sentinel = np.arange(4, dtype=np.float32)

    def fake_dequantize(_data, _tensor_type):
        return sentinel

    actual = _dequantize_gguf_tensor(np.zeros((1, 18), dtype=np.uint8), FakeQuantType("Q4_0", 2), fake_dequantize)
    assert actual is sentinel
