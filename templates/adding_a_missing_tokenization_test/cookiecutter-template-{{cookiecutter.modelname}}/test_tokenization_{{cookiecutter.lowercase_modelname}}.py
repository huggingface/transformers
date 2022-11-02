# coding=utf-8
# Copyright 2022 {{cookiecutter.authors}}. All rights reserved.
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
""" Testing suite for the {{cookiecutter.modelname}} tokenizer. """


import unittest

{% if cookiecutter.has_slow_class == "True" and  cookiecutter.has_fast_class == "True" -%}
from transformers import {{cookiecutter.camelcase_modelname}}Tokenizer, {{cookiecutter.camelcase_modelname}}TokenizerFast
{% elif  cookiecutter.has_slow_class == "True" -%}
from transformers import {{cookiecutter.camelcase_modelname}}Tokenizer
{% elif  cookiecutter.has_fast_class == "True" -%}
from transformers import {{cookiecutter.camelcase_modelname}}TokenizerFast
{% endif -%}
{% if cookiecutter.has_fast_class == "True" and  cookiecutter.slow_tokenizer_use_sentencepiece == "True" -%}
from transformers.testing_utils import require_sentencepiece, require_tokenizers
from ...test_tokenization_common import TokenizerTesterMixin


@require_sentencepiece
@require_tokenizers
{% elif  cookiecutter.slow_tokenizer_use_sentencepiece == "True" -%}
from transformers.testing_utils import require_sentencepiece
from ...test_tokenization_common import TokenizerTesterMixin


@require_sentencepiece
{% elif  cookiecutter.has_fast_class == "True" -%}
from transformers.testing_utils import require_tokenizers
from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
{% else -%}
from ...test_tokenization_common import TokenizerTesterMixin


{% endif -%}
class {{cookiecutter.camelcase_modelname}}TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    {% if cookiecutter.has_slow_class == "True" -%}
    tokenizer_class = {{cookiecutter.camelcase_modelname}}Tokenizer
    test_slow_tokenizer = True
    {% else -%}
    tokenizer_class = None
    test_slow_tokenizer = False
    {% endif -%}
    {% if cookiecutter.has_fast_class == "True" -%}
    rust_tokenizer_class = {{cookiecutter.camelcase_modelname}}TokenizerFast
    test_rust_tokenizer = True
    {% else -%}
    rust_tokenizer_class = None
    test_rust_tokenizer = False
    {% endif -%}
    {% if  cookiecutter.slow_tokenizer_use_sentencepiece == "True" -%}
    test_sentencepiece = True
    {% endif -%}
    # TODO: Check in `TokenizerTesterMixin` if other attributes need to be changed
    def setUp(self):
        super().setUp()

        raise NotImplementedError(
            "Here you have to implement the saving of a toy tokenizer in "
            "`self.tmpdirname`."
        )

    # TODO: add tests with hard-coded target values 