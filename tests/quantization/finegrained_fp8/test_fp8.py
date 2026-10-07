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
"""The deprecated `FineGrainedFP8Config`: passed explicitly, it loads through its own frozen integration; a checkpoint
declaring `quant_method="fp8"` loads as [`BlockFp8Config`]. The formats it spelled are tested in `blockfp8` and
`mxfp8`."""

import unittest
import warnings

from transformers import PreTrainedConfig
from transformers.testing_utils import require_accelerate, require_torch
from transformers.utils.quantization_config import BlockFp8Config, FineGrainedFP8Config


def _legacy(**options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return FineGrainedFP8Config(**options)


def _quantizer(passed, checkpoint=None):
    """The quantizer `from_pretrained` builds for a passed config, over a checkpoint's config if any."""
    from transformers.quantizers.auto import get_hf_quantizer

    model_config = PreTrainedConfig()
    if checkpoint is not None:
        model_config.quantization_config = dict(checkpoint)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        quantizer, _, _ = get_hf_quantizer(model_config, passed, None, True, {})
    return quantizer


@require_torch
@require_accelerate
class FineGrainedFP8LegacyTest(unittest.TestCase):
    def test_building_one_is_deprecated(self):
        with self.assertWarns(FutureWarning):
            FineGrainedFP8Config()

    def test_a_passed_deprecated_config_loads_through_its_frozen_integration(self):
        """On the fly and over an `fp8` checkpoint alike: the checkpoint's config is read as the passed one's class."""
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

        for checkpoint in (None, {"quant_method": "fp8", "weight_block_size": [128, 128]}):
            with self.subTest(pre_quantized=checkpoint is not None):
                quantizer = _quantizer(_legacy(dequantize=True), checkpoint)
                self.assertIsInstance(quantizer, FineGrainedFP8HfQuantizer)
                self.assertIsInstance(quantizer.quantization_config, FineGrainedFP8Config)
                self.assertTrue(quantizer.quantization_config.dequantize)

    def test_an_fp8_checkpoint_loads_as_block_fp8(self):
        from transformers.quantizers.quantizer_blockfp8 import BlockFp8HfQuantizer

        quantizer = _quantizer(None, {"quant_method": "fp8", "weight_block_size": [128, 128]})
        self.assertIsInstance(quantizer, BlockFp8HfQuantizer)
        self.assertIsInstance(quantizer.quantization_config, BlockFp8Config)
