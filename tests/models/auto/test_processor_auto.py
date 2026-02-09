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
from pathlib import Path
from shutil import copyfile

from huggingface_hub import snapshot_download, upload_folder

import transformers
from transformers import (
    CONFIG_MAPPING,
    FEATURE_EXTRACTOR_MAPPING,
    MODEL_FOR_AUDIO_TOKENIZATION_MAPPING,
    PROCESSOR_MAPPING,
    TOKENIZER_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoTokenizer,
    BaseVideoProcessor,
    BertTokenizer,
    CLIPImageProcessorFast,
    FeatureExtractionMixin,
    ImageProcessingMixin,
    LlamaTokenizer,
    LlavaOnevisionVideoProcessor,
    LlavaProcessor,
    ProcessorMixin,
    SiglipImageProcessor,
    SiglipImageProcessorFast,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)
from transformers.models.auto.feature_extraction_auto import get_feature_extractor_config
from transformers.models.auto.image_processing_auto import get_image_processor_config
from transformers.models.auto.video_processing_auto import get_video_processor_config
from transformers.testing_utils import TOKEN, TemporaryHubRepo, get_tests_dir, is_staging_test
from transformers.tokenization_python import TOKENIZER_CONFIG_FILE
from transformers.utils import (
    FEATURE_EXTRACTOR_NAME,
    PROCESSOR_NAME,
)


sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402
from test_module.custom_feature_extraction import CustomFeatureExtractor  # noqa E402
from test_module.custom_processing import CustomProcessor  # noqa E402
from test_module.custom_tokenization import CustomTokenizer  # noqa E402


SAMPLE_PROCESSOR_CONFIG = get_tests_dir("fixtures/dummy_feature_extractor_config.json")
SAMPLE_VOCAB_LLAMA = get_tests_dir("fixtures/test_sentencepiece.model")
SAMPLE_VOCAB = get_tests_dir("fixtures/vocab.json")
SAMPLE_CONFIG = get_tests_dir("fixtures/config.json")
SAMPLE_PROCESSOR_CONFIG_DIR = get_tests_dir("fixtures")


class AutoFeatureExtractorTest(unittest.TestCase):
    vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "bla", "blou"]

    def setUp(self):
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    def test_processor_from_model_shortcut(self):
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_repo(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config()
            processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

            # save in new folder
            model_config.save_pretrained(tmpdirname)
            processor.save_pretrained(tmpdirname)

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_subfolder_from_repo(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            processor.save_pretrained(f"{tmpdirname}/processor_subfolder")

            processor = Wav2Vec2Processor.from_pretrained(tmpdirname, subfolder="processor_subfolder")

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_extractor_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # copy relevant files
            copyfile(SAMPLE_PROCESSOR_CONFIG, os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME))
            copyfile(SAMPLE_VOCAB, os.path.join(tmpdirname, "vocab.json"))
            copyfile(SAMPLE_CONFIG, os.path.join(tmpdirname, "config.json"))

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_subcomponent_get_config_dict_saved_as_nested_config(self):
        """
        Tests that we can get config dict of a subcomponents of a processor,
        even if they were saved as nested dict in `processor_config.json`
        """
        # Test feature extractor first
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            processor.save_pretrained(tmpdirname)

            config_dict_1 = get_feature_extractor_config(tmpdirname)
            feature_extractor_1 = Wav2Vec2FeatureExtractor(**config_dict_1)
            self.assertIsInstance(feature_extractor_1, Wav2Vec2FeatureExtractor)

            config_dict_2, _ = FeatureExtractionMixin.get_feature_extractor_dict(tmpdirname)
            feature_extractor_2 = Wav2Vec2FeatureExtractor(**config_dict_2)
            self.assertIsInstance(feature_extractor_2, Wav2Vec2FeatureExtractor)
            self.assertEqual(config_dict_1, config_dict_2)

        # Test image and video processors next
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
            processor.save_pretrained(tmpdirname)

            config_dict_1 = get_image_processor_config(tmpdirname)
            image_processor_1 = SiglipImageProcessor(**config_dict_1)
            self.assertIsInstance(image_processor_1, SiglipImageProcessor)

            config_dict_2, _ = ImageProcessingMixin.get_image_processor_dict(tmpdirname)
            image_processor_2 = SiglipImageProcessor(**config_dict_2)
            self.assertIsInstance(image_processor_2, SiglipImageProcessor)
            self.assertEqual(config_dict_1, config_dict_2)

            config_dict_1 = get_video_processor_config(tmpdirname)
            video_processor_1 = LlavaOnevisionVideoProcessor(**config_dict_1)
            self.assertIsInstance(video_processor_1, LlavaOnevisionVideoProcessor)

            config_dict_2, _ = BaseVideoProcessor.get_video_processor_dict(tmpdirname)
            video_processor_2 = LlavaOnevisionVideoProcessor(**config_dict_2)
            self.assertIsInstance(video_processor_2, LlavaOnevisionVideoProcessor)
            self.assertEqual(config_dict_1, config_dict_2)

    def test_processor_from_processor_class(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            feature_extractor = Wav2Vec2FeatureExtractor()
            tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

            processor = Wav2Vec2Processor(feature_extractor, tokenizer)

            # save in new folder
            processor.save_pretrained(tmpdirname)

            if not os.path.isfile(os.path.join(tmpdirname, PROCESSOR_NAME)):
                # create one manually in order to perform this test's objective
                config_dict = {"processor_class": "Wav2Vec2Processor"}
                with open(os.path.join(tmpdirname, PROCESSOR_NAME), "w") as fp:
                    json.dump(config_dict, fp)

            # drop `processor_class` in tokenizer config
            with open(os.path.join(tmpdirname, TOKENIZER_CONFIG_FILE)) as f:
                config_dict = json.load(f)
                config_dict.pop("processor_class")

            with open(os.path.join(tmpdirname, TOKENIZER_CONFIG_FILE), "w") as f:
                f.write(json.dumps(config_dict))

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_tokenizer_processor_class(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            feature_extractor = Wav2Vec2FeatureExtractor()
            tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

            processor = Wav2Vec2Processor(feature_extractor, tokenizer)

            # save in new folder
            processor.save_pretrained(tmpdirname)

            # drop `processor_class` in processor
            with open(os.path.join(tmpdirname, PROCESSOR_NAME)) as f:
                config_dict = json.load(f)
                config_dict.pop("processor_class")
            with open(os.path.join(tmpdirname, PROCESSOR_NAME), "w") as f:
                f.write(json.dumps(config_dict))

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_model_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config(processor_class="Wav2Vec2Processor")
            model_config.save_pretrained(tmpdirname)
            # copy relevant files
            copyfile(SAMPLE_VOCAB, os.path.join(tmpdirname, "vocab.json"))
            # create empty sample processor
            with open(os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME), "w") as f:
                f.write("{}")

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_from_pretrained_dynamic_processor(self):
        # If remote code is not set, we will time out when asking whether to load the model.
        with self.assertRaises(ValueError):
            processor = AutoProcessor.from_pretrained("hf-internal-testing/test_dynamic_processor_updated")
        # If remote code is disabled, we can't load this config.
        with self.assertRaises(ValueError):
            processor = AutoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_processor_updated", trust_remote_code=False
            )

        processor = AutoProcessor.from_pretrained(
            "hf-internal-testing/test_dynamic_processor_updated", trust_remote_code=True
        )
        self.assertTrue(processor.special_attribute_present)
        self.assertEqual(processor.__class__.__name__, "NewProcessor")

        feature_extractor = processor.feature_extractor
        self.assertTrue(feature_extractor.special_attribute_present)
        self.assertEqual(feature_extractor.__class__.__name__, "NewFeatureExtractor")

        tokenizer = processor.tokenizer
        self.assertTrue(tokenizer.special_attribute_present)
        self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")

        new_processor = AutoProcessor.from_pretrained(
            "hf-internal-testing/test_dynamic_processor", trust_remote_code=True, use_fast=False
        )
        new_tokenizer = new_processor.tokenizer
        self.assertTrue(new_tokenizer.special_attribute_present)
        self.assertEqual(new_tokenizer.__class__.__name__, "NewTokenizerFast")

    def test_new_processor_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, CustomFeatureExtractor)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            AutoProcessor.register(CustomConfig, CustomProcessor)
            # Trying to register something existing in the Transformers library will raise an error
            with self.assertRaises(ValueError):
                AutoProcessor.register(Wav2Vec2Config, Wav2Vec2Processor)

            # Now that the config is registered, it can be used as any other config with the auto-API
            feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)

            with tempfile.TemporaryDirectory() as tmp_dir:
                vocab_file = os.path.join(tmp_dir, "vocab.txt")
                with open(vocab_file, "w", encoding="utf-8") as vocab_writer:
                    vocab_writer.write("".join([x + "\n" for x in self.vocab_tokens]))
                tokenizer = CustomTokenizer(vocab_file)

            processor = CustomProcessor(feature_extractor, tokenizer)

            with tempfile.TemporaryDirectory() as tmp_dir:
                processor.save_pretrained(tmp_dir)
                new_processor = AutoProcessor.from_pretrained(tmp_dir)
                self.assertIsInstance(new_processor, CustomProcessor)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            if CustomConfig in PROCESSOR_MAPPING._extra_content:
                del PROCESSOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in MODEL_FOR_AUDIO_TOKENIZATION_MAPPING._extra_content:
                del MODEL_FOR_AUDIO_TOKENIZATION_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_processor_conflict(self):
        class NewFeatureExtractor(Wav2Vec2FeatureExtractor):
            special_attribute_present = False

        class NewTokenizer(BertTokenizer):
            special_attribute_present = False

        class NewProcessor(ProcessorMixin):
            special_attribute_present = False

            def __init__(self, feature_extractor, tokenizer):
                super().__init__(feature_extractor, tokenizer)

        try:
            AutoConfig.register("custom", CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, NewFeatureExtractor)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=NewTokenizer)
            AutoProcessor.register(CustomConfig, NewProcessor)
            # If remote code is not set, the default is to use local classes.
            processor = AutoProcessor.from_pretrained("hf-internal-testing/test_dynamic_processor_updated")
            self.assertEqual(processor.__class__.__name__, "NewProcessor")
            self.assertFalse(processor.special_attribute_present)
            self.assertFalse(processor.feature_extractor.special_attribute_present)
            self.assertFalse(processor.tokenizer.special_attribute_present)

            # If remote code is disabled, we load the local ones.
            processor = AutoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_processor_updated", trust_remote_code=False
            )
            self.assertEqual(processor.__class__.__name__, "NewProcessor")
            self.assertFalse(processor.special_attribute_present)
            self.assertFalse(processor.feature_extractor.special_attribute_present)
            self.assertFalse(processor.tokenizer.special_attribute_present)

            # If remote is enabled, we load from the Hub.
            processor = AutoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_processor_updated", trust_remote_code=True
            )
            self.assertEqual(processor.__class__.__name__, "NewProcessor")
            self.assertTrue(processor.special_attribute_present)
            self.assertTrue(processor.feature_extractor.special_attribute_present)
            self.assertTrue(processor.tokenizer.special_attribute_present)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            if CustomConfig in PROCESSOR_MAPPING._extra_content:
                del PROCESSOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in MODEL_FOR_AUDIO_TOKENIZATION_MAPPING._extra_content:
                del MODEL_FOR_AUDIO_TOKENIZATION_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_processor_with_extra_attributes(self):
        class NewFeatureExtractor(Wav2Vec2FeatureExtractor):
            pass

        class NewTokenizer(BertTokenizer):
            pass

        class NewProcessor(ProcessorMixin):
            def __init__(self, feature_extractor, tokenizer, processor_attr_1=1, processor_attr_2=True):
                super().__init__(feature_extractor, tokenizer)

                self.processor_attr_1 = processor_attr_1
                self.processor_attr_2 = processor_attr_2

        try:
            AutoConfig.register("custom", CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, NewFeatureExtractor)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=NewTokenizer)
            AutoProcessor.register(CustomConfig, NewProcessor)
            # If remote code is not set, the default is to use local classes.
            processor = AutoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_processor_updated", processor_attr_2=False
            )
            self.assertEqual(processor.__class__.__name__, "NewProcessor")
            self.assertEqual(processor.processor_attr_1, 1)
            self.assertEqual(processor.processor_attr_2, False)
        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            if CustomConfig in PROCESSOR_MAPPING._extra_content:
                del PROCESSOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in MODEL_FOR_AUDIO_TOKENIZATION_MAPPING._extra_content:
                del MODEL_FOR_AUDIO_TOKENIZATION_MAPPING._extra_content[CustomConfig]

    def test_dynamic_processor_with_specific_dynamic_subcomponents(self):
        class NewFeatureExtractor(Wav2Vec2FeatureExtractor):
            pass

        class NewTokenizer(BertTokenizer):
            pass

        class NewProcessor(ProcessorMixin):
            def __init__(self, feature_extractor, tokenizer):
                super().__init__(feature_extractor, tokenizer)

        try:
            AutoConfig.register("custom", CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, NewFeatureExtractor)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=NewTokenizer)
            AutoProcessor.register(CustomConfig, NewProcessor)
            # If remote code is not set, the default is to use local classes.
            processor = AutoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_processor_updated",
            )
            self.assertEqual(processor.__class__.__name__, "NewProcessor")
        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            if CustomConfig in PROCESSOR_MAPPING._extra_content:
                del PROCESSOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in MODEL_FOR_AUDIO_TOKENIZATION_MAPPING._extra_content:
                del MODEL_FOR_AUDIO_TOKENIZATION_MAPPING._extra_content[CustomConfig]

    def test_auto_processor_creates_tokenizer(self):
        processor = AutoProcessor.from_pretrained("hf-internal-testing/tiny-random-bert")
        self.assertEqual(processor.__class__.__name__, "BertTokenizer")

    def test_auto_processor_creates_image_processor(self):
        processor = AutoProcessor.from_pretrained("hf-internal-testing/tiny-random-convnext")
        self.assertEqual(processor.__class__.__name__, "ConvNextImageProcessorFast")

    def test_auto_processor_save_load(self):
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            second_processor = AutoProcessor.from_pretrained(tmp_dir)
            self.assertEqual(second_processor.__class__.__name__, processor.__class__.__name__)

    def test_processor_with_multiple_tokenizers_save_load(self):
        """Test that processors with multiple tokenizers save and load correctly."""

        class DualTokenizerProcessor(ProcessorMixin):
            """A processor with two tokenizers and an image processor."""

            def __init__(self, tokenizer, decoder_tokenizer, image_processor):
                super().__init__(tokenizer, decoder_tokenizer, image_processor)

        # Create processor with multiple tokenizers
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BertForMaskedLM")
        decoder_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        image_processor = SiglipImageProcessor()

        processor = DualTokenizerProcessor(
            tokenizer=tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            image_processor=image_processor,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)

            # Verify directory structure: primary tokenizer in root, additional in subfolder
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "tokenizer_config.json")))
            self.assertTrue(os.path.isdir(os.path.join(tmp_dir, "decoder_tokenizer")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "decoder_tokenizer", "tokenizer_config.json")))

            # Verify processor_config.json contains image_processor but not tokenizers
            with open(os.path.join(tmp_dir, "processor_config.json")) as f:
                processor_config = json.load(f)
            self.assertIn("image_processor", processor_config)
            self.assertNotIn("tokenizer", processor_config)
            self.assertNotIn("decoder_tokenizer", processor_config)

            # Reload the full processor and verify all attributes
            loaded_processor = DualTokenizerProcessor.from_pretrained(tmp_dir)

            # Verify the processor has all expected attributes
            self.assertTrue(hasattr(loaded_processor, "tokenizer"))
            self.assertTrue(hasattr(loaded_processor, "decoder_tokenizer"))
            self.assertTrue(hasattr(loaded_processor, "image_processor"))

            # Verify tokenizers loaded correctly
            self.assertEqual(loaded_processor.tokenizer.vocab_size, tokenizer.vocab_size)
            self.assertEqual(loaded_processor.decoder_tokenizer.vocab_size, decoder_tokenizer.vocab_size)

            # Verify image processor loaded correctly
            self.assertEqual(loaded_processor.image_processor.size, image_processor.size)

    def test_processor_with_multiple_image_processors_save_load(self):
        """Test that processors with multiple image processors save and load correctly."""

        class DualImageProcessorProcessor(ProcessorMixin):
            """A processor with two image processors and a tokenizer."""

            def __init__(self, tokenizer, image_processor, encoder_image_processor):
                super().__init__(tokenizer, image_processor, encoder_image_processor)

        # Create processor with multiple image processors
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BertForMaskedLM")
        image_processor = SiglipImageProcessorFast(size={"height": 224, "width": 224})
        encoder_image_processor = CLIPImageProcessorFast(size={"height": 384, "width": 384})

        processor = DualImageProcessorProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            encoder_image_processor=encoder_image_processor,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)

            # Verify processor_config.json contains both image processors
            with open(os.path.join(tmp_dir, "processor_config.json")) as f:
                processor_config = json.load(f)
            self.assertIn("image_processor", processor_config)
            self.assertIn("encoder_image_processor", processor_config)
            self.assertNotIn("tokenizer", processor_config)

            # Verify both image processors have the correct type key for instantiation
            self.assertIn("image_processor_type", processor_config["image_processor"])
            self.assertIn("image_processor_type", processor_config["encoder_image_processor"])
            self.assertEqual(processor_config["image_processor"]["image_processor_type"], "SiglipImageProcessorFast")
            self.assertEqual(
                processor_config["encoder_image_processor"]["image_processor_type"], "CLIPImageProcessorFast"
            )

            # Verify the sizes are different (to ensure they're separate configs)
            self.assertEqual(processor_config["image_processor"]["size"], {"height": 224, "width": 224})
            self.assertEqual(processor_config["encoder_image_processor"]["size"], {"height": 384, "width": 384})

            # Reload the full processor and verify all attributes
            loaded_processor = DualImageProcessorProcessor.from_pretrained(tmp_dir)

            # Verify the processor has all expected attributes
            self.assertTrue(hasattr(loaded_processor, "tokenizer"))
            self.assertTrue(hasattr(loaded_processor, "image_processor"))
            self.assertTrue(hasattr(loaded_processor, "encoder_image_processor"))

            # Verify tokenizer loaded correctly
            self.assertEqual(loaded_processor.tokenizer.vocab_size, tokenizer.vocab_size)

            # Verify image processors loaded correctly with their distinct sizes
            self.assertEqual(loaded_processor.image_processor.size, {"height": 224, "width": 224})
            self.assertEqual(loaded_processor.encoder_image_processor.size, {"height": 384, "width": 384})

            # Verify they are different types
            self.assertIsInstance(loaded_processor.image_processor, SiglipImageProcessorFast)
            self.assertIsInstance(loaded_processor.encoder_image_processor, CLIPImageProcessorFast)


@is_staging_test
class ProcessorPushToHubTester(unittest.TestCase):
    vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "bla", "blou"]

    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN

    def test_push_to_hub_via_save_pretrained(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            processor = Wav2Vec2Processor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                processor.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_processor = Wav2Vec2Processor.from_pretrained(tmp_repo.repo_id)
            for k, v in processor.feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_processor.feature_extractor, k))
            self.assertDictEqual(new_processor.tokenizer.get_vocab(), processor.tokenizer.get_vocab())

    def test_push_to_hub_in_organization_via_save_pretrained(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            processor = Wav2Vec2Processor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                processor.save_pretrained(
                    tmp_dir,
                    repo_id=tmp_repo.repo_id,
                    push_to_hub=True,
                    token=self._token,
                )

            new_processor = Wav2Vec2Processor.from_pretrained(tmp_repo.repo_id)
            for k, v in processor.feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_processor.feature_extractor, k))
            self.assertDictEqual(new_processor.tokenizer.get_vocab(), processor.tokenizer.get_vocab())

    def test_push_to_hub_dynamic_processor(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            CustomFeatureExtractor.register_for_auto_class()
            CustomTokenizer.register_for_auto_class()
            CustomProcessor.register_for_auto_class()

            feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)

            with tempfile.TemporaryDirectory() as tmp_dir:
                vocab_file = os.path.join(tmp_dir, "vocab.txt")
                with open(vocab_file, "w", encoding="utf-8") as vocab_writer:
                    vocab_writer.write("".join([x + "\n" for x in self.vocab_tokens]))
                tokenizer = CustomTokenizer(vocab_file)

            processor = CustomProcessor(feature_extractor, tokenizer)

            with tempfile.TemporaryDirectory() as tmp_dir:
                snapshot_download(tmp_repo.repo_id, token=self._token)
                processor.save_pretrained(tmp_dir)

                # This has added the proper auto_map field to the feature extractor config
                self.assertDictEqual(
                    processor.feature_extractor.auto_map,
                    {
                        "AutoFeatureExtractor": "custom_feature_extraction.CustomFeatureExtractor",
                        "AutoProcessor": "custom_processing.CustomProcessor",
                    },
                )

                # This has added the proper auto_map field to the tokenizer config
                with open(os.path.join(tmp_dir, "tokenizer_config.json")) as f:
                    tokenizer_config = json.load(f)
                self.assertDictEqual(
                    tokenizer_config["auto_map"],
                    {
                        "AutoTokenizer": ["custom_tokenization.CustomTokenizer", None],
                        "AutoProcessor": "custom_processing.CustomProcessor",
                    },
                )

                # The code has been copied from fixtures
                self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "custom_feature_extraction.py")))
                self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "custom_tokenization.py")))
                self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "custom_processing.py")))

                upload_folder(repo_id=tmp_repo.repo_id, folder_path=tmp_dir, token=self._token)

                new_processor = AutoProcessor.from_pretrained(tmp_repo.repo_id, trust_remote_code=True)
                # Can't make an isinstance check because the new_processor is from the CustomProcessor class of a dynamic module
                self.assertEqual(new_processor.__class__.__name__, "CustomProcessor")

    def test_push_to_hub_with_chat_templates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer = LlamaTokenizer.from_pretrained(SAMPLE_VOCAB_LLAMA)
            image_processor = SiglipImageProcessor()
            chat_template = "default dummy template for testing purposes only"
            processor = LlavaProcessor(
                tokenizer=tokenizer, image_processor=image_processor, chat_template=chat_template
            )
            self.assertEqual(processor.chat_template, chat_template)
            with TemporaryHubRepo(token=self._token) as tmp_repo:
                processor.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, token=self._token, push_to_hub=True)
                reloaded_processor = LlavaProcessor.from_pretrained(tmp_repo.repo_id)
                self.assertEqual(processor.chat_template, reloaded_processor.chat_template)
                # When we save as single files, tokenizers and processors share a chat template, which means
                # the reloaded tokenizer should get the chat template as well
                self.assertEqual(reloaded_processor.chat_template, reloaded_processor.tokenizer.chat_template)

            with TemporaryHubRepo(token=self._token) as tmp_repo:
                processor.chat_template = {"default": "a", "secondary": "b"}
                processor.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, token=self._token, push_to_hub=True)
                reloaded_processor = LlavaProcessor.from_pretrained(tmp_repo.repo_id)
                self.assertEqual(processor.chat_template, reloaded_processor.chat_template)
                # When we save as single files, tokenizers and processors share a chat template, which means
                # the reloaded tokenizer should get the chat template as well
                self.assertEqual(reloaded_processor.chat_template, reloaded_processor.tokenizer.chat_template)
