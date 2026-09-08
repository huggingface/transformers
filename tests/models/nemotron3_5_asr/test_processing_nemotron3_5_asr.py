# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from transformers import (
    AutoProcessor,
    Nemotron3_5AsrProcessor,
    NemotronAsrStreamingAudioProcessor,
    PreTrainedTokenizerFast,
)
from transformers.testing_utils import require_torch, require_torchaudio


@require_torch
@require_torchaudio
class NemotronAudioProcessorMigrationTest(unittest.TestCase):
    def setUp(self):
        self.audio_processor = NemotronAsrStreamingAudioProcessor()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<blank>": 1}, unk_token="<unk>")),
            unk_token="<unk>",
            pad_token="<blank>",
        )

    def make_processor(self):
        return Nemotron3_5AsrProcessor(audio_processor=self.audio_processor, tokenizer=self.tokenizer)

    def test_legacy_constructor_and_attribute_alias(self):
        with self.assertWarnsRegex(FutureWarning, "audio_processor"):
            processor = Nemotron3_5AsrProcessor(feature_extractor=self.audio_processor, tokenizer=self.tokenizer)
        self.assertIs(processor.audio_processor, self.audio_processor)
        with self.assertWarnsRegex(FutureWarning, "audio_processor"):
            self.assertIs(processor.feature_extractor, self.audio_processor)
        replacement = NemotronAsrStreamingAudioProcessor()
        with self.assertWarnsRegex(FutureWarning, "audio_processor"):
            processor.feature_extractor = replacement
        self.assertIs(processor.audio_processor, replacement)
        self.assertEqual(processor.get_attributes(), ["audio_processor", "tokenizer"])
        self.assertNotIn("feature_extractor", processor.to_dict())
        with self.assertWarns(FutureWarning):
            processor = Nemotron3_5AsrProcessor(
                audio_processor=self.audio_processor, feature_extractor=replacement, tokenizer=self.tokenizer
            )
        self.assertIs(processor.audio_processor, self.audio_processor)
        self.assertIs(
            Nemotron3_5AsrProcessor(self.audio_processor, self.tokenizer).audio_processor, self.audio_processor
        )

    def test_config_migration_and_audio_output(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), TemporaryDirectory() as directory:
                processor = self.make_processor()
                processor.save_pretrained(directory)
                config_path = Path(directory) / "processor_config.json"
                config = json.loads(config_path.read_text())
                self.assertIn("audio_processor", config)
                self.assertNotIn("feature_extractor", config)
                if legacy:
                    config["feature_extractor"] = config.pop("audio_processor")
                    config_path.write_text(json.dumps(config))
                with warnings.catch_warnings(record=True) as caught:
                    restored = AutoProcessor.from_pretrained(directory)
                self.assertFalse(any("AutoFeatureExtractor" in str(w.message) for w in caught))
                self.assertEqual(restored.audio_processor.to_dict(), processor.audio_processor.to_dict())
                audio = np.zeros(16000, dtype=np.float32)
                expected = processor.audio_processor(audio, sampling_rate=16000, return_tensors="np")
                actual = restored.audio_processor(audio, sampling_rate=16000, return_tensors="np")
                self.assertEqual(actual.keys(), expected.keys())
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])
                restored.save_pretrained(directory)
                saved = json.loads(config_path.read_text())
                self.assertIn("audio_processor", saved)
                self.assertNotIn("feature_extractor", saved)

    def test_canonical_config_wins_over_legacy(self):
        with TemporaryDirectory() as directory:
            self.make_processor().save_pretrained(directory)
            config_path = Path(directory) / "processor_config.json"
            config = json.loads(config_path.read_text())
            config["feature_extractor"] = {**config["audio_processor"], "sampling_rate": 8000}
            config_path.write_text(json.dumps(config))
            restored = Nemotron3_5AsrProcessor.from_pretrained(directory)
            self.assertEqual(restored.audio_processor.sampling_rate, 16000)

    def test_audio_kwargs_use_canonical_component(self):
        from transformers.processing_utils import ProcessingKwargs

        processor = self.make_processor()
        with warnings.catch_warnings(record=True) as caught:
            kwargs = processor._merge_kwargs(ProcessingKwargs, audio_kwargs={"return_padding_mask": False})
        self.assertFalse(caught)
        self.assertIs(kwargs["audio_kwargs"]["return_padding_mask"], False)
