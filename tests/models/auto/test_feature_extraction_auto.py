# Copyright 2021 the HuggingFace Inc. team.
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
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import transformers
from transformers import (
    CONFIG_MAPPING,
    FEATURE_EXTRACTOR_MAPPING,
    AutoAudioProcessor,
    AutoConfig,
    AutoFeatureExtractor,
    Wav2Vec2AudioProcessor,
    Wav2Vec2AudioProcessorNumpy,
    Wav2Vec2Config,
    WhisperAudioProcessor,
    WhisperAudioProcessorNumpy,
)
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER, get_tests_dir


sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402
from test_module.custom_feature_extraction import CustomFeatureExtractor  # noqa E402


SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR = get_tests_dir("fixtures")
SAMPLE_FEATURE_EXTRACTION_CONFIG = get_tests_dir("fixtures/dummy_feature_extractor_config.json")
SAMPLE_CONFIG = get_tests_dir("fixtures/dummy-config.json")


def _write_processor_config(dirname, config_dict):
    with open(Path(dirname) / "preprocessor_config.json", "w") as fp:
        json.dump(config_dict, fp)


class AutoAudioProcessorTest(unittest.TestCase):
    def setUp(self):
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    def test_audio_processor_from_model_shortcut(self):
        audio_processor = AutoAudioProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsInstance(audio_processor, Wav2Vec2AudioProcessor)

    def test_audio_processor_from_local_directory_from_key(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _write_processor_config(tmpdirname, {"audio_processor_type": "WhisperAudioProcessor"})
            audio_processor = AutoAudioProcessor.from_pretrained(tmpdirname)
            self.assertIsInstance(audio_processor, WhisperAudioProcessor)

    def test_audio_processor_from_local_directory_from_legacy_feature_extractor_key(self):
        # Legacy hub configs identify the class with `feature_extractor_type`
        audio_processor = AutoAudioProcessor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
        self.assertIsInstance(audio_processor, Wav2Vec2AudioProcessor)

    def test_audio_processor_from_legacy_irregular_name(self):
        # Legacy names that don't follow the FeatureExtractor -> AudioProcessor substitution
        # resolve through LEGACY_FEATURE_EXTRACTOR_NAME_MAP
        with tempfile.TemporaryDirectory() as tmpdirname:
            _write_processor_config(tmpdirname, {"feature_extractor_type": "ASTFeatureExtractor"})
            audio_processor = AutoAudioProcessor.from_pretrained(tmpdirname)
            self.assertEqual(type(audio_processor).__name__, "AudioSpectrogramTransformerAudioProcessor")

    def test_audio_processor_from_local_file(self):
        audio_processor = AutoAudioProcessor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG)
        self.assertIsInstance(audio_processor, Wav2Vec2AudioProcessor)

    def test_audio_processor_from_local_directory_from_model_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config()

            # remove audio_processor_type to make sure config.json alone is enough to load the
            # audio processor locally
            config_dict = AutoAudioProcessor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR).to_dict()
            config_dict.pop("audio_processor_type")
            audio_processor = Wav2Vec2AudioProcessor(**config_dict)

            # save in new folder
            model_config.save_pretrained(tmpdirname)
            audio_processor.save_pretrained(tmpdirname)
            # drop the type key written by save_pretrained so resolution goes through the model config
            saved = json.load(open(Path(tmpdirname) / "preprocessor_config.json"))
            saved.pop("audio_processor_type", None)
            _write_processor_config(tmpdirname, saved)

            audio_processor = AutoAudioProcessor.from_pretrained(tmpdirname)

            # make sure private variable is not incorrectly saved
            dict_as_saved = json.loads(audio_processor.to_json_string())
            self.assertTrue("_processor_class" not in dict_as_saved)

        self.assertIsInstance(audio_processor, Wav2Vec2AudioProcessor)

    # ── Backend selection ──────────────────────────────────────────────────────────

    def test_backend_kwarg_torch(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _write_processor_config(tmpdirname, {"audio_processor_type": "WhisperAudioProcessor"})
            audio_processor = AutoAudioProcessor.from_pretrained(tmpdirname, backend="torch")
            self.assertIsInstance(audio_processor, WhisperAudioProcessor)

    def test_backend_kwarg_numpy(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _write_processor_config(tmpdirname, {"audio_processor_type": "WhisperAudioProcessor"})
            audio_processor = AutoAudioProcessor.from_pretrained(tmpdirname, backend="numpy")
            self.assertIsInstance(audio_processor, WhisperAudioProcessorNumpy)

    def test_backend_kwarg_numpy_from_legacy_name(self):
        # Backend selection composes with legacy-name translation
        audio_processor = AutoAudioProcessor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR, backend="numpy")
        self.assertIsInstance(audio_processor, Wav2Vec2AudioProcessorNumpy)

    def test_backend_fallback_warns_when_sibling_missing(self):
        # phi4_multimodal has no numpy sibling: requesting numpy falls back to torch
        with tempfile.TemporaryDirectory() as tmpdirname:
            _write_processor_config(tmpdirname, {"audio_processor_type": "Phi4MultimodalAudioProcessor"})
            audio_processor = AutoAudioProcessor.from_pretrained(tmpdirname, backend="numpy")
            self.assertEqual(type(audio_processor).__name__, "Phi4MultimodalAudioProcessor")

    def test_invalid_backend_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown audio-processor backend"):
            AutoAudioProcessor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR, backend="jax")

    # ── Error paths ────────────────────────────────────────────────────────────────

    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, "bert-base is not a local folder and is not a valid model identifier"
        ):
            _ = AutoAudioProcessor.from_pretrained("bert-base")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = AutoAudioProcessor.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")

    def test_audio_processor_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError,
            "Can't load audio processor for 'hf-internal-testing/config-no-model'.",
        ):
            _ = AutoAudioProcessor.from_pretrained("hf-internal-testing/config-no-model")

    def test_unrecognized_audio_processor_error(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _write_processor_config(tmpdirname, {"audio_processor_type": "DoesNotExistAudioProcessor"})
            with self.assertRaisesRegex(ValueError, "Unrecognized audio processor"):
                AutoAudioProcessor.from_pretrained(tmpdirname)

    # ── Registration ───────────────────────────────────────────────────────────────

    def test_new_audio_processor_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            AutoAudioProcessor.register(CustomConfig, CustomFeatureExtractor)
            # Trying to register something existing in the Transformers library will raise an error
            with self.assertRaises(ValueError):
                AutoAudioProcessor.register(Wav2Vec2Config, Wav2Vec2AudioProcessor)

            # Now that the config is registered, it can be used as any other config with the auto-API
            audio_processor = CustomFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            with tempfile.TemporaryDirectory() as tmp_dir:
                audio_processor.save_pretrained(tmp_dir)
                new_audio_processor = AutoAudioProcessor.from_pretrained(tmp_dir)
                self.assertIsInstance(new_audio_processor, CustomFeatureExtractor)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]

    def test_register_with_backend_dict(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            AutoAudioProcessor.register(CustomConfig, {"torch": CustomFeatureExtractor})

            audio_processor = CustomFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            with tempfile.TemporaryDirectory() as tmp_dir:
                audio_processor.save_pretrained(tmp_dir)
                # numpy backend has no sibling registered: falls back to the torch entry
                new_audio_processor = AutoAudioProcessor.from_pretrained(tmp_dir, backend="numpy")
                self.assertIsInstance(new_audio_processor, CustomFeatureExtractor)
        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]

    # ── Remote code ────────────────────────────────────────────────────────────────

    def test_from_pretrained_dynamic_audio_processor(self):
        # If remote code is not set, we will time out when asking whether to load the model.
        with self.assertRaises(ValueError):
            _ = AutoAudioProcessor.from_pretrained("hf-internal-testing/test_dynamic_feature_extractor")
        # If remote code is disabled, we can't load this config.
        with self.assertRaises(ValueError):
            _ = AutoAudioProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=False
            )

        audio_processor = AutoAudioProcessor.from_pretrained(
            "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=True
        )
        self.assertEqual(audio_processor.__class__.__name__, "NewFeatureExtractor")

        # Test the dynamic module is loaded only once.
        reloaded_audio_processor = AutoAudioProcessor.from_pretrained(
            "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=True
        )
        self.assertIs(audio_processor.__class__, reloaded_audio_processor.__class__)

        # Test the audio processor can be reloaded.
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_processor.save_pretrained(tmp_dir)
            reloaded_audio_processor = AutoAudioProcessor.from_pretrained(tmp_dir, trust_remote_code=True)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "feature_extractor.py")))  # Assert we saved code
        self.assertEqual(reloaded_audio_processor.__class__.__name__, "NewFeatureExtractor")

    def test_from_pretrained_dynamic_audio_processor_conflict(self):
        class NewFeatureExtractor(Wav2Vec2AudioProcessor):
            is_local = True

        try:
            AutoConfig.register("custom", CustomConfig)
            AutoAudioProcessor.register(CustomConfig, NewFeatureExtractor)
            # If remote code is not set, the default is to use local
            audio_processor = AutoAudioProcessor.from_pretrained("hf-internal-testing/test_dynamic_feature_extractor")
            self.assertEqual(audio_processor.__class__.__name__, "NewFeatureExtractor")
            self.assertTrue(audio_processor.is_local)

            # If remote code is disabled, we load the local one.
            audio_processor = AutoAudioProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=False
            )
            self.assertEqual(audio_processor.__class__.__name__, "NewFeatureExtractor")
            self.assertTrue(audio_processor.is_local)

            # If remote code is enabled but the user explicitly registered the local one, we load the local one.
            audio_processor = AutoAudioProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=True
            )
            self.assertEqual(audio_processor.__class__.__name__, "NewFeatureExtractor")
            self.assertTrue(audio_processor.is_local)

            # If remote code is enabled but local code originated from transformers, we load the remote one.
            NewFeatureExtractor.__module__ = "transformers.models.custom.configuration_custom"
            audio_processor = AutoAudioProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=True
            )
            self.assertEqual(audio_processor.__class__.__name__, "NewFeatureExtractor")
            self.assertTrue(not hasattr(audio_processor, "is_local"))

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]


class AutoFeatureExtractorBackwardCompatTest(unittest.TestCase):
    """`AutoFeatureExtractor` is a deprecated alias for `AutoAudioProcessor` (removal: v5.15);
    it must warn and resolve to the same classes until then."""

    def test_from_pretrained_warns_and_resolves(self):
        with self.assertWarns(FutureWarning):
            audio_processor = AutoFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
        self.assertIsInstance(audio_processor, Wav2Vec2AudioProcessor)

    def test_from_pretrained_matches_auto_audio_processor(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            legacy = AutoFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
        modern = AutoAudioProcessor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
        self.assertIs(legacy.__class__, modern.__class__)
        self.assertEqual(legacy.to_dict(), modern.to_dict())

    def test_backend_kwarg_passthrough(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            audio_processor = AutoFeatureExtractor.from_pretrained(
                SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR, backend="numpy"
            )
        self.assertIsInstance(audio_processor, Wav2Vec2AudioProcessorNumpy)

    def test_register_warns_and_registers(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            with self.assertWarns(FutureWarning):
                AutoFeatureExtractor.register(CustomConfig, CustomFeatureExtractor)

            audio_processor = CustomFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            with tempfile.TemporaryDirectory() as tmp_dir:
                audio_processor.save_pretrained(tmp_dir)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    new_audio_processor = AutoFeatureExtractor.from_pretrained(tmp_dir)
                self.assertIsInstance(new_audio_processor, CustomFeatureExtractor)
        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]
