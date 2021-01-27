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

import unittest

from transformers import AutoConfig, AutoTokenizer, BertConfig, TensorType, is_flax_available
from transformers.testing_utils import require_flax, slow


if is_flax_available():
    import jax
    from transformers.models.auto.modeling_flax_auto import FlaxAutoModel
    from transformers.models.bert.modeling_flax_bert import FlaxBertModel
    from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaModel


@require_flax
class FlaxAutoModelTest(unittest.TestCase):
    @slow
    def test_bert_from_pretrained(self):
        for model_name in ["bert-base-cased", "bert-large-uncased"]:
            with self.subTest(model_name):
                config = AutoConfig.from_pretrained(model_name)
                self.assertIsNotNone(config)
                self.assertIsInstance(config, BertConfig)

                model = FlaxAutoModel.from_pretrained(model_name)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, FlaxBertModel)

    @slow
    def test_roberta_from_pretrained(self):
        for model_name in ["roberta-base-cased", "roberta-large-uncased"]:
            with self.subTest(model_name):
                config = AutoConfig.from_pretrained(model_name)
                self.assertIsNotNone(config)
                self.assertIsInstance(config, BertConfig)

                model = FlaxAutoModel.from_pretrained(model_name)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, FlaxRobertaModel)

    @slow
    def test_bert_jax_jit(self):
        for model_name in ["bert-base-cased", "bert-large-uncased"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = FlaxBertModel.from_pretrained(model_name)
            tokens = tokenizer("Do you support jax jitted function?", return_tensors=TensorType.JAX)

            @jax.jit
            def eval(**kwargs):
                return model(**kwargs)

            eval(**tokens).block_until_ready()

    @slow
    def test_roberta_jax_jit(self):
        for model_name in ["roberta-base-cased", "roberta-large-uncased"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = FlaxRobertaModel.from_pretrained(model_name)
            tokens = tokenizer("Do you support jax jitted function?", return_tensors=TensorType.JAX)

            @jax.jit
            def eval(**kwargs):
                return model(**kwargs)

            eval(**tokens).block_until_ready()
