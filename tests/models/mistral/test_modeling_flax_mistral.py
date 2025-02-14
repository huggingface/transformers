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

from transformers import MistralConfig, is_flax_available, is_tokenizers_available
from transformers.testing_utils import require_flax, slow

from ...generation.test_flax_utils import FlaxGenerationTesterMixin
from ...test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor


if is_flax_available():
    import jax.numpy as jnp

    from transformers.models.mistral.modeling_flax_mistral import (
        FlaxMistralForCausalLM,
        FlaxMistralModel,
    )


if is_tokenizers_available():
    from transformers import LlamaTokenizerFast


class FlaxMistralModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        window_size=7,
        initializer_range=0.02,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.window_size = window_size
        self.initializer_range = initializer_range
        self.scope = None
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = np.tril(np.ones((self.batch_size, self.seq_length)))

        config = MistralConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=True,
            is_decoder=False,
            initializer_range=self.initializer_range,
            sliding_window=self.window_size,
        )
        config.pad_token_id = config.eos_token_id

        return (config, input_ids, input_mask)

    # Copied from tests.models.gpt_neo.test_modeling_flax_gpt_neo.FlaxGPTNeoModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict

    # Copied from tests.models.gpt_neo.test_modeling_flax_gpt_neo.FlaxGPTNeoModelTester.check_use_cache_forward
    def check_use_cache_forward(self, model_class_name, config, input_ids, attention_mask):
        max_decoder_length = 20
        model = model_class_name(config)

        past_key_values = model.init_cache(input_ids.shape[0], max_decoder_length)
        attention_mask = jnp.ones((input_ids.shape[0], max_decoder_length), dtype="i4")

        position_ids = jnp.broadcast_to(
            jnp.arange(input_ids.shape[-1] - 1)[None, :], (input_ids.shape[0], input_ids.shape[-1] - 1)
        )
        outputs_cache = model(
            input_ids[:, :-1],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        position_ids = jnp.array(input_ids.shape[0] * [[input_ids.shape[-1] - 1]], dtype="i4")
        outputs_cache_next = model(
            input_ids[:, -1:],
            attention_mask=attention_mask,
            past_key_values=outputs_cache.past_key_values,
            position_ids=position_ids,
        )

        outputs = model(input_ids)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")

    # Copied from tests.models.gpt_neo.test_modeling_flax_gpt_neo.FlaxGPTNeoModelTester.check_use_cache_forward_with_attn_mask
    def check_use_cache_forward_with_attn_mask(self, model_class_name, config, input_ids, attention_mask):
        max_decoder_length = 20
        model = model_class_name(config)

        attention_mask_cache = jnp.concatenate(
            [attention_mask, jnp.zeros((attention_mask.shape[0], max_decoder_length - attention_mask.shape[1]))],
            axis=-1,
        )

        past_key_values = model.init_cache(input_ids.shape[0], max_decoder_length)
        position_ids = jnp.broadcast_to(
            jnp.arange(input_ids.shape[-1] - 1)[None, :], (input_ids.shape[0], input_ids.shape[-1] - 1)
        )

        outputs_cache = model(
            input_ids[:, :-1],
            attention_mask=attention_mask_cache,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_ids = jnp.array(input_ids.shape[0] * [[input_ids.shape[-1] - 1]], dtype="i4")
        outputs_cache_next = model(
            input_ids[:, -1:],
            past_key_values=outputs_cache.past_key_values,
            attention_mask=attention_mask_cache,
            position_ids=position_ids,
        )

        outputs = model(input_ids, attention_mask=attention_mask)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")


@require_flax
class FlaxMistralModelTest(FlaxModelTesterMixin, FlaxGenerationTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxMistralModel, FlaxMistralForCausalLM) if is_flax_available() else ()

    def setUp(self):
        self.model_tester = FlaxMistralModelTester(self)

    def test_use_cache_forward(self):
        for model_class_name in self.all_model_classes:
            config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
            self.model_tester.check_use_cache_forward(model_class_name, config, input_ids, attention_mask)

    def test_use_cache_forward_with_attn_mask(self):
        for model_class_name in self.all_model_classes:
            config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
            self.model_tester.check_use_cache_forward_with_attn_mask(
                model_class_name, config, input_ids, attention_mask
            )

    @slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("mistralai/Mistral-7B-v0.1", from_pt=True)
            outputs = model(np.ones((1, 1)))
            self.assertIsNotNone(outputs)


@slow
@require_flax
class FlaxMistralIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "mistralai/Mistral-7B-v0.1"
        self.model = FlaxMistralForCausalLM.from_pretrained(self.model_id, from_pt=True)
        self.test_batch = jnp.arange(32).reshape(4, 8) + 1911

    def test_model_logits(self):
        input_ids = jnp.array([[1, 306, 4658, 278, 6593, 310, 2834, 338]])
        EXPECTED_MEAN = np.array([[-2.5548, -2.5737, -3.0600, -2.5906, -2.8478, -2.8118, -2.9325, -2.7694]])
        EXPECTED_SLICE = np.array([-5.8781,-5.8616,-0.1052,-4.7200,-5.8781,-5.8774,-5.8773,-5.8777,-5.8781,-5.8780,-5.8781,-5.8779,-1.0787,1.7583,-5.8779,-5.8780,-5.8783,-5.8778,-5.8776,-5.8781,-5.8784,-5.8778,-5.8778,-5.8777,-5.8779,-5.8778,-5.8776,-5.8780,-5.8779,-5.8781])  # fmt: skip

        flax_logits = self.model(input_ids).logits
        diff_mean = jnp.abs(flax_logits.mean(-1) - EXPECTED_MEAN).max()
        diff_slice = jnp.abs(flax_logits[0, 0, :30] - EXPECTED_SLICE).max()

        self.assertAlmostEqual(diff_mean, 0, places=3)
        self.assertAlmostEqual(diff_slice, 0, places=3)

    def test_generated_text(self):
        tokenizer = LlamaTokenizerFast.from_pretrained(self.model_id)
        tokenizer.pad_token_id = 2
        EXPECTED_TEXT_COMPLETION = """My favourite condiment is 100% ketchup. I love it on everything. I’m not a big"""
        prompt = "My favourite condiment is "
        inputs = tokenizer(prompt, return_tensors="np", truncation=True, padding=True)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20, temperature=0).sequences
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT_COMPLETION)
