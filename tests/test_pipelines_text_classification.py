# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import importlib
import logging
import string
import unittest
from functools import lru_cache

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TOKENIZER_MAPPING,
    AutoTokenizer,
    TextClassificationPipeline,
)


logger = logging.getLogger(__name__)


def get_checkpoint_from_architecture(architecture):
    module = importlib.import_module(architecture.__module__)

    if hasattr(module, "_CHECKPOINT_FOR_DOC"):
        return module._CHECKPOINT_FOR_DOC
    else:
        logger.warning(f"Can't retrieve checkpoint from {architecture.__name__}")


def get_tiny_config_from_class(configuration_class):
    if "OpenAIGPT" in configuration_class.__name__:
        # This is the only file that is inconsistent with the naming scheme.
        # Will rename this file if we decide this is the way to go
        return

    model_type = configuration_class.model_type
    camel_case_model_name = configuration_class.__name__.split("Config")[0]

    module = importlib.import_module(f".test_modeling_{model_type.replace('-', '_')}", package="tests")
    model_tester_class = getattr(module, f"{camel_case_model_name}ModelTester", None)

    if model_tester_class is None:
        logger.warning(f"No model tester class for {configuration_class.__name__}")
        return

    model_tester = model_tester_class(parent=None)

    if hasattr(model_tester, "get_config"):
        return model_tester.get_config()
    else:
        logger.warning(f"Model tester {model_tester_class.__name__} has no `get_config()`.")


@lru_cache(maxsize=100)
def get_tiny_tokenizer_from_checkpoint(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    logger.warning("Training new from iterator ...")
    vocabulary = string.ascii_letters + string.digits + " "
    tokenizer = tokenizer.train_new_from_iterator(vocabulary, vocab_size=len(vocabulary))
    logger.warning("Trained.")
    return tokenizer


class TextClassificationPipelineTests(unittest.TestCase):
    pipeline = TextClassificationPipeline
    model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

    @classmethod
    def setUpClass(cls) -> None:
        model_and_tokenizers = ()

        for configuration, model_architecture in cls.model_mapping.items():
            print(configuration.__name__)
            tokenizer_classes = TOKENIZER_MAPPING.get(configuration, [])
            checkpoint = get_checkpoint_from_architecture(model_architecture)
            tiny_config = get_tiny_config_from_class(configuration)

            if tiny_config is None or checkpoint is None:
                continue

            logger.warning(f"Building model from {tiny_config.__class__.__name__}")
            model = model_architecture(tiny_config)
            logger.warning("Built.")

            for tokenizer_class in tokenizer_classes:
                if tokenizer_class is not None and tokenizer_class.__name__.endswith("Fast"):
                    model_and_tokenizers += ((model, get_tiny_tokenizer_from_checkpoint(checkpoint)),)
        cls.model_and_tokenizers = model_and_tokenizers

    def test_inference_single_input(self):
        for model, tokenizer in self.model_and_tokenizers:
            with self.subTest(f"Testing {model.__class__.__name__} with {tokenizer.__class__.__name__}"):
                text_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

                valid_inputs = "HuggingFace is in Paris and Dumbo"
                outputs = text_classifier(valid_inputs)

                self.assertIsInstance(outputs, list)
                self.assertEqual(len(outputs), 1)

                single_output = outputs[0]
                self.assertListEqual(list(single_output.keys()), ["label", "score"])
                self.assertTrue(single_output["label"] in model.config.id2label.values())

    def test_inference_multi_input(self):
        for model, tokenizer in self.model_and_tokenizers:
            with self.subTest(f"Testing {model.__class__.__name__} with {tokenizer.__class__.__name__}"):
                text_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

                valid_inputs = ["HuggingFace is in Paris", "HuggingFace is in Dumbo"]
                outputs = text_classifier(valid_inputs)
                self.assertIsInstance(outputs, list)
                self.assertEqual(len(outputs), len(valid_inputs))

                for single_output in outputs:
                    self.assertListEqual(list(single_output.keys()), ["label", "score"])
                    self.assertTrue(single_output["label"] in model.config.id2label.values())
