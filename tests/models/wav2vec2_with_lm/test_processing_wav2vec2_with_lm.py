# Copyright 2026 HuggingFace Inc.
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
"""`Wav2Vec2ProcessorWithLM` is deprecated: only the deprecation and missing-dependency paths are tested, without
pyctcdecode / kenlm installed."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from transformers.models.wav2vec2_with_lm import Wav2Vec2ProcessorWithLM
from transformers.pipelines import _maybe_load_ctc_decoder
from transformers.utils import import_utils


class Wav2Vec2ProcessorWithLMDeprecationTest(unittest.TestCase):
    def test_from_pretrained_raises_without_dependencies(self):
        missing = {
            "pyctcdecode": (lambda: False, import_utils.PYCTCDECODE_IMPORT_ERROR),
            "kenlm": (lambda: False, import_utils.KENLM_IMPORT_ERROR),
        }
        with mock.patch.dict(import_utils.BACKENDS_MAPPING, missing):
            with self.assertRaises(ImportError) as context:
                Wav2Vec2ProcessorWithLM.from_pretrained("hf-internal-testing/processor_with_lm")

        message = str(context.exception)
        self.assertIn("pip install pyctcdecode kenlm", message)
        self.assertIn("Transformers v6", message)
        self.assertIn("Wav2Vec2Processor", message)
        self.assertIn("#migrating-to-pyctcdecode", message)

    def test_pipeline_falls_back_to_greedy_with_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "preprocessor_config.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "feature_extractor_type": "Wav2Vec2FeatureExtractor",
                        "processor_class": "Wav2Vec2ProcessorWithLM",
                    },
                    f,
                )

            kwargs = {}
            # `None` in `sys.modules` makes `import kenlm` raise an `ImportError`
            with mock.patch.dict(sys.modules, {"kenlm": None}):
                with self.assertWarns(UserWarning) as context:
                    _maybe_load_ctc_decoder(tmpdir, {}, kwargs, None)

        self.assertNotIn("decoder", kwargs)
        message = str(context.warning)
        self.assertIn("falls back to greedy CTC", message)
        self.assertIn("pip install pyctcdecode kenlm", message)
        self.assertIn("Transformers v6", message)
