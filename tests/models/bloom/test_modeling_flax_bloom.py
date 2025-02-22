# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import numpy as np

from transformers import BloomConfig, BloomTokenizerFast, is_flax_available
from transformers.testing_utils import require_flax, slow

from ...test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor


if is_flax_available():
    import os

    # The slow tests are often failing with OOM error on GPU
    # This makes JAX allocate exactly what is needed on demand, and deallocate memory that is no longer needed
    # but will be slower as stated here https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax.numpy as jnp

    from transformers import FlaxBloomForCausalLM, FlaxBloomModel


def prepare_bloom_inputs_dict(config, input_ids, attention_mask=None):
    if attention_mask is None:
        attention_mask = np.where(input_ids != config.pad_token_id, 1, 0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@require_flax
class FlaxBloomModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        n_layer=2,
        n_head=4,
        hidden_act="gelu",
        hidden_dropout=0.1,
        attention_probs_dropout_prob=0.1,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        initializer_range=0.02,
        apply_residual_connection_post_layernorm=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.initializer_range = initializer_range
        self.is_encoder_decoder = False
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm

    def prepare_config_and_inputs(self):
        input_ids = np.clip(ids_tensor([self.batch_size, self.seq_length - 1], self.vocab_size), 3, self.vocab_size)
        input_ids = np.concatenate((input_ids, 2 * np.ones((self.batch_size, 1), dtype=np.int64)), -1)

        config = BloomConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_probs_dropout_prob,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            is_encoder_decoder=False,
            use_cache=False,
        )
        inputs_dict = prepare_bloom_inputs_dict(config, input_ids)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def check_use_cache_forward(self, model_class_name, config, inputs_dict):
        max_length = 20
        model = model_class_name(config)

        input_ids = inputs_dict["input_ids"]
        attention_mask = jnp.ones((input_ids.shape[0], max_length), dtype="i4")

        past_key_values = model.init_cache(input_ids.shape[0], max_length)

        outputs_cache = model(
            input_ids[:, :-1],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        outputs_cache_next = model(
            input_ids[:, -1:],
            attention_mask=attention_mask,
            past_key_values=outputs_cache.past_key_values,
        )

        outputs = model(input_ids)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")

    def check_use_cache_forward_with_attn_mask(self, model_class_name, config, inputs_dict):
        max_length = 20
        model = model_class_name(config)

        input_ids, attention_mask = (
            inputs_dict["input_ids"],
            inputs_dict["attention_mask"],
        )

        attention_mask_cache = jnp.concatenate(
            [
                attention_mask,
                jnp.zeros((attention_mask.shape[0], max_length - attention_mask.shape[1])),
            ],
            axis=-1,
        )

        past_key_values = model.init_cache(input_ids.shape[0], max_length)

        outputs_cache = model(
            input_ids[:, :-1],
            attention_mask=attention_mask_cache,
            past_key_values=past_key_values,
        )
        outputs_cache_next = model(
            input_ids[:, -1:],
            past_key_values=outputs_cache.past_key_values,
            attention_mask=attention_mask_cache,
        )

        outputs = model(input_ids, attention_mask=attention_mask)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")


@require_flax
class FlaxBloomModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxBloomModel, FlaxBloomForCausalLM) if is_flax_available() else ()

    def setUp(self):
        self.model_tester = FlaxBloomModelTester(self)

    def test_use_cache_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            self.model_tester.check_use_cache_forward(model_class, config, inputs_dict)

    def test_use_cache_forward_with_attn_mask(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            self.model_tester.check_use_cache_forward_with_attn_mask(model_class, config, inputs_dict)

    @slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("bigscience/bloom-560m")
            input_ids = np.ones((1, 1)) * model.config.eos_token_id
            outputs = model(input_ids)
            self.assertIsNotNone(outputs)


@slow
@require_flax
class FlaxBloomGenerationTest(unittest.TestCase):
    all_model_classes = (FlaxBloomForCausalLM,) if is_flax_available() else ()

    def setUp(self):
        self.model_id = "bigscience/bloom-560m"
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.model_id, padding_side="left")
        self.model_tester = FlaxBloomModelTester(self)
        self.model = FlaxBloomForCausalLM.from_pretrained(self.model_id, from_pt=True, revision="gs555750")

    def test_model_batched_gen(self):
        # tests if the model outputs the same generation for the same batched input
        input_sentences = [
            "Hello there is this string is definitely longer I believe that",
            "Hello there is this string is definitely longer I believe that",
        ]
        inputs = self.tokenizer(input_sentences, return_tensors="np", padding=True, truncation=True)
        sequences_fx = self.model.generate(**inputs, max_length=20).sequences
        self.assertEqual(sequences_fx[0].tolist(), sequences_fx[1].tolist())

    def test_model_batched_padding_left(self):
        # tests if the model outputs the same generation for an input that is part of a batch
        # and a single input
        input_sentences_batch = [
            "Hello there is this string is definitely longer I believe that",
            "Hi I want to order",
        ]
        inputs = self.tokenizer(input_sentences_batch, return_tensors="np", padding=True, truncation=True)
        sequences_fx_batch = self.model.generate(**inputs, max_length=20).sequences

        input_sentence_simple = "Hi I want to order"
        inputs_simple = self.tokenizer(input_sentence_simple, return_tensors="np")
        sequences_fx_simple = self.model.generate(**inputs_simple, max_length=20).sequences

        self.assertEqual(sequences_fx_batch[1][6:].tolist(), sequences_fx_simple[0][:-6].tolist())

    def test_batch_generated_text(self):
        input_sentences = [
            "Hello what is",
            "Running a quick test with the",
        ]
        inputs = self.tokenizer(input_sentences, return_tensors="np", padding=True, truncation=True)
        generated_ids = self.model.generate(**inputs, max_length=20).sequences
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # these generations match those of the PyTorch model, ensuring correctness
        EXPECTED_GENERATIONS = [
            "Hello what is the best way to get the data from the server? I have tried",
            "Running a quick test with the following command:\nsudo apt-get install python3\nsudo apt-get install python2",
        ]

        self.assertListEqual(generated_text, EXPECTED_GENERATIONS)
