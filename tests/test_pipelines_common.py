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
from functools import lru_cache
from unittest import skipIf

from transformers import FEATURE_EXTRACTOR_MAPPING, TOKENIZER_MAPPING, AutoFeatureExtractor, AutoTokenizer


logger = logging.getLogger(__name__)


def get_checkpoint_from_architecture(architecture):
    try:
        module = importlib.import_module(architecture.__module__)
    except ImportError:
        logger.error(f"Ignoring architecture {architecture}")
        return

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

    try:
        module = importlib.import_module(f".test_modeling_{model_type.replace('-', '_')}", package="tests")
        model_tester_class = getattr(module, f"{camel_case_model_name}ModelTester", None)
    except (ImportError, AttributeError):
        logger.error(f"No model tester class for {configuration_class.__name__}")
        return

    if model_tester_class is None:
        logger.warning(f"No model tester class for {configuration_class.__name__}")
        return

    model_tester = model_tester_class(parent=None)

    if hasattr(model_tester, "get_pipeline_config"):
        return model_tester.get_pipeline_config()
    elif hasattr(model_tester, "get_config"):
        return model_tester.get_config()
    else:
        logger.warning(f"Model tester {model_tester_class.__name__} has no `get_config()`.")


@lru_cache(maxsize=100)
def get_tiny_tokenizer_from_checkpoint(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    logger.info("Training new from iterator ...")
    vocabulary = string.ascii_letters + string.digits + " "
    tokenizer = tokenizer.train_new_from_iterator(vocabulary, vocab_size=len(vocabulary), show_progress=False)
    logger.info("Trained.")
    return tokenizer


def get_tiny_feature_extractor_from_checkpoint(checkpoint, tiny_config):
    feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
    feature_extractor = feature_extractor.__class__(size=tiny_config.image_size, crop_size=tiny_config.image_size)
    return feature_extractor


class ANY:
    def __init__(self, _type):
        self._type = _type

    def __eq__(self, other):
        return isinstance(other, self._type)

    def __repr__(self):
        return f"ANY({self._type.__name__})"


class PipelineTestCaseMeta(type):
    def __new__(mcs, name, bases, dct):
        def gen_tokenizer_test(ModelClass, checkpoint, tiny_config, tokenizer_class):
            @skipIf(tiny_config is None, "TinyConfig does not exist")
            @skipIf(checkpoint is None, "checkpoint does not exist")
            def test(self):
                model = ModelClass(tiny_config)
                if hasattr(model, "eval"):
                    model = model.eval()
                try:
                    tokenizer = get_tiny_tokenizer_from_checkpoint(checkpoint)
                    if hasattr(model.config, "max_position_embeddings"):
                        tokenizer.model_max_length = model.config.max_position_embeddings
                # Rust Panic exception are NOT Exception subclass
                # Some test tokenizer contain broken vocabs or custom PreTokenizer, so we
                # provide some default tokenizer and hope for the best.
                except:  # noqa: E722
                    self.skipTest(f"Ignoring {ModelClass}, cannot create a simple tokenizer")
                self.run_pipeline_test(model, tokenizer)

            return test

        def gen_feature_test(ModelClass, checkpoint, tiny_config, feature_extractor_class):
            @skipIf(tiny_config is None, "TinyConfig does not exist")
            @skipIf(checkpoint is None, "checkpoint does not exist")
            def test(self):
                model = ModelClass(tiny_config)
                if hasattr(model, "eval"):
                    model = model.eval()
                feature_extractor = get_tiny_feature_extractor_from_checkpoint(checkpoint, tiny_config)
                self.run_pipeline_test(model, feature_extractor)

            return test

        for prefix, key in [("pt", "model_mapping"), ("tf", "tf_model_mapping")]:
            mapping = dct.get(key, {})
            if mapping:
                for configuration, model_architectures in mapping.items():
                    if not isinstance(model_architectures, tuple):
                        model_architectures = (model_architectures,)

                    for model_architecture in model_architectures:
                        checkpoint = get_checkpoint_from_architecture(model_architecture)
                        tiny_config = get_tiny_config_from_class(configuration)
                        tokenizer_classes = TOKENIZER_MAPPING.get(configuration, [])
                        for tokenizer_class in tokenizer_classes:
                            if tokenizer_class is not None and tokenizer_class.__name__.endswith("Fast"):
                                test_name = f"test_{prefix}_{configuration.__name__}_{model_architecture.__name__}_{tokenizer_class.__name__}"
                                dct[test_name] = gen_tokenizer_test(
                                    model_architecture, checkpoint, tiny_config, tokenizer_class
                                )

                        feature_extractor_class = FEATURE_EXTRACTOR_MAPPING.get(configuration, None)
                        if feature_extractor_class is not None:
                            test_name = f"test_{prefix}_{configuration.__name__}_{model_architecture.__name__}_{feature_extractor_class.__name__}"
                            dct[test_name] = gen_feature_test(
                                model_architecture, checkpoint, tiny_config, feature_extractor_class
                            )

        return type.__new__(mcs, name, bases, dct)
