# Copyright 2025 HuggingFace Inc.
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
import unittest

from transformers import AlbertConfig
from transformers.validators import interval, probability, token, activation_fn_key
from huggingface_hub.dataclasses import StrictDataclassFieldValidationError


class ValidatorsTests(unittest.TestCase):
    """
    Sanity check tests for the validators. Note that the validators do not perform strict type checking
    (`huggingface_hub.dataclasses.strict` is used for that).
    """

    def test_interval(self):
        # valid
        interval(1, 10)(5)
        interval(1, 10)(5.0)
        interval(1, 10)(10)
        interval(1, 10)(1)
        interval(1, 10, exclude_min=True)(1.0000001)

        # invalid
        with self.assertRaises(ValueError):
            interval(1, 10)(11)  # greater than max
        with self.assertRaises(ValueError):
            interval(1, 10)(0.9999999)  # less than min
        with self.assertRaises(ValueError):
            interval(1, 10, exclude_max=True)(10)  # equal to max, but exclude_max is True
        with self.assertRaises(ValueError):
            interval(1, 10, exclude_min=True)(1.0)  # equal to min, but exclude_min is True
        with self.assertRaises(ValueError):
            interval(1, 10)(-5)  # less than min

    def test_probability(self):
        # valid
        probability(0.5)
        probability(0)
        probability(1)

        # invalid
        with self.assertRaises(ValueError):
            probability(99)  # 0-1 probabilities only
        with self.assertRaises(ValueError):
            probability(1.1)  # greater than 1
        with self.assertRaises(ValueError):
            probability(-0.1)  # less than 0

    def test_token(self):
        # valid
        token(None)
        token(0)
        token(1)
        token(999999999)

        # invalid
        with self.assertRaises(ValueError):
            token(-1)  # less than 0
        with self.assertRaises(TypeError):
            token("<eos>")  # must be the token id, not its string counterpart

    def test_activation_fn_key(self):
        # valid
        activation_fn_key("relu")
        activation_fn_key("gelu")

        # invalid
        with self.assertRaises(ValueError):
            activation_fn_key("foo")  # obvious one
        with self.assertRaises(ValueError):
            activation_fn_key(None)  # can't be None
        with self.assertRaises(ValueError):
            activation_fn_key("Relu")  # typo: should be "relu", not "Relu"


class ValidatorsIntegrationTests(unittest.TestCase):
    """Tests in which the validators are used as part of another class/function"""

    def test_model_config_validation(self):
        """Sanity check tests for the integration of model config with `huggingface_hub.dataclasses.strict`"""
        # 1 - We can initialize the config, including with arbitrary kwargs
        config = AlbertConfig()
        config = AlbertConfig(eos_token_id=5)
        self.assertEqual(config.eos_token_id, 5)
        config = AlbertConfig(eos_token_id=None)
        self.assertIsNone(config.eos_token_id)
        config = AlbertConfig(foo="bar")  # Ensures backwards compatibility
        self.assertEqual(config.foo, "bar")

        # 2 - Manual specification, traveling through an invalid config, should be allowed
        config.eos_token_id = 99  # vocab_size = 30000, eos_token_id = 99 -> valid
        config.vocab_size = 10  # vocab_size = 10, eos_token_id = 99 -> invalid (but only throws error in `validate()`)
        with self.assertRaises(ValueError):
            config.validate()
        config.eos_token_id = 9  # vocab_size = 10, eos_token_id = 9 -> valid
        config.validate()

        # 3 - These cases should raise an error

        # vocab_size is an int
        with self.assertRaises(StrictDataclassFieldValidationError):
            config = AlbertConfig(vocab_size=10.0)

        # num_hidden_layers is an int
        with self.assertRaises(StrictDataclassFieldValidationError):
            config = AlbertConfig(num_hidden_layers=None)

        # position_embedding_type is a Literal, foo is not one of the options
        with self.assertRaises(StrictDataclassFieldValidationError):
            config = AlbertConfig(position_embedding_type="foo")

       # eos_token_id is a token, and must be non-negative
        with self.assertRaises(StrictDataclassFieldValidationError):
            config = AlbertConfig(eos_token_id=-1)

        # `validate()` is called in `__post_init__`, i.e. after `__init__`. A special token must be in the vocabulary.
        with self.assertRaises(ValueError):
            config = AlbertConfig(vocab_size=10, eos_token_id=99)

        # vocab size is assigned after init, individual attributes are checked on assignment
        with self.assertRaises(StrictDataclassFieldValidationError):
            config = AlbertConfig()
            config.vocab_size = "foo"
