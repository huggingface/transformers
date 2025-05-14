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
from dataclasses import dataclass
from typing import Optional, Union

from huggingface_hub.dataclasses import StrictDataclassFieldValidationError, strict

from transformers import AlbertConfig
from transformers.validators import activation_fn_key, interval, probability, token


class ValidatorsTests(unittest.TestCase):
    """
    Sanity check tests for the validators. Validators are `field` in a dataclass, and not meant to be used on
    their own.
    """

    def test_interval(self):
        # Setup test dataclasses
        @strict
        @dataclass
        class TestInterval:
            data: Union[int, float] = interval(min=1, max=10)()

        @strict
        @dataclass
        class TestIntervalExcludeMinMax:
            data: Union[int, float] = interval(min=1, max=10, exclude_min=True, exclude_max=True)()

        # valid
        TestInterval(5)
        TestInterval(5.0)
        TestInterval(10)
        TestInterval(1)
        TestIntervalExcludeMinMax(1.0000001)

        # invalid
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestInterval("one")  # different type
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestInterval(11)  # greater than max
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestInterval(0.9999999)  # less than min
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestIntervalExcludeMinMax(10)  # equal to max, but exclude_max is True
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestIntervalExcludeMinMax(1.0)  # equal to min, but exclude_min is True
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestInterval(-5)  # less than min

    def test_probability(self):
        # Setup test dataclasses
        @strict
        @dataclass
        class TestProbability:
            data: float = probability()

        # valid
        TestProbability(0.5)
        TestProbability(0.0)
        TestProbability(1.0)

        # invalid
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestProbability(1)  # different type
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestProbability(99.0)  # 0-1 probabilities only
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestProbability(1.1)  # greater than 1
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestProbability(-0.1)  # less than 0

    def test_token(self):
        # Setup test dataclasses
        @strict
        @dataclass
        class TestToken:
            data: Optional[int] = token()

        # valid
        TestToken(None)
        TestToken(0)
        TestToken(1)
        TestToken(999999999)

        # invalid
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestToken(-1)  # less than 0
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestToken("<eos>")  # different type: must be the token id, not its string counterpart

    def test_activation_fn_key(self):
        # Setup test dataclasses
        @strict
        @dataclass
        class TestActivationFnKey:
            data: str = activation_fn_key()

        # valid
        TestActivationFnKey("relu")
        TestActivationFnKey("gelu")

        # invalid
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestActivationFnKey("foo")  # obvious one
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestActivationFnKey(None)  # different type: can't be None
        with self.assertRaises(StrictDataclassFieldValidationError):
            TestActivationFnKey("Relu")  # typo: should be "relu", not "Relu"


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
