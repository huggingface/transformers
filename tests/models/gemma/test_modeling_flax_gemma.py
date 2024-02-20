# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import jax
import numpy as np

from transformers import AutoTokenizer, GemmaConfig, is_flax_available, is_tokenizers_available
from transformers.testing_utils import require_flax, slow

from ...generation.test_flax_utils import FlaxGenerationTesterMixin
from ...test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor


if is_flax_available():
    import jax.numpy as jnp

    from transformers.models.gemma.modeling_flax_gemma import (
        FlaxGemmaForCausalLM,
        FlaxGemmaModel,
    )


if is_tokenizers_available():
    pass


class FlaxGemmaModelTester:
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

        config = GemmaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.hidden_size // self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=True,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

        return config, input_ids, input_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict

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
class FlaxGemmaModelTest(FlaxModelTesterMixin, FlaxGenerationTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxGemmaModel, FlaxGemmaForCausalLM) if is_flax_available() else ()
    all_generative_model_classes = (FlaxGemmaForCausalLM,) if is_flax_available() else ()

    def setUp(self):
        self.model_tester = FlaxGemmaModelTester(self)

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
            model = model_class_name.from_pretrained("openlm-research/open_gemma_3b_v2", from_pt=True)
            outputs = model(np.ones((1, 1)))
            self.assertIsNotNone(outputs)


@slow
@require_flax
class FlaxGemmaIntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]

    def test_model_2b_fp32(self):
        # TODO: change it to the new repo after the release
        model_id = "gg-hf/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 198-199 Ford Mustang GT. I am trying to",
            "Hi today I am going to show you how to make a simple and easy to make a simple and easy to",
        ]

        model, params = FlaxGemmaForCausalLM.from_pretrained(model_id, revision="flax", _do_init=False)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="np", padding=True)

        output = model.generate(**inputs, params=params, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

        jit_generate = jax.jit(model.generate)
        output_sequences = jit_generate(**inputs).sequences
        output_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_2b_fp16(self):
        # TODO: change it to the new repo after the release
        model_id = "gg-hf/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 198-199 Ford Mustang GT. I am trying to",
            "Hi today I am going to show you how to make a simple and easy to make a simple and easy to",
        ]

        model, params = FlaxGemmaForCausalLM.from_pretrained(model_id, revision="flax", _do_init=False, dtype=jnp.float16)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="np", padding=True)

        output = model.generate(**inputs, params=params, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

        jit_generate = jax.jit(model.generate)
        output_sequences = jit_generate(**inputs).sequences
        output_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_2b_bf16(self):
        # TODO: change it to the new repo after the release
        model_id = "gg-hf/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 198-199 Ford Mustang GT. I am trying to",
            "Hi today I am going to show you how to make a simple and easy to make a simple and easy to",
        ]

        model, params = FlaxGemmaForCausalLM.from_pretrained(model_id, revision="flax", _do_init=False, dtype=jnp.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="np", padding=True)

        output = model.generate(**inputs, params=params, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

        jit_generate = jax.jit(model.generate)
        output_sequences = jit_generate(**inputs).sequences
        output_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)

    @unittest.skip("The test will not fit our CI runners")
    def test_model_7b_fp32(self):
        # TODO: change it to the new repo after the release
        model_id = "gg-hf/gemma-7b"
        EXPECTED_TEXTS = [
            """Hello I am doing a project on the topic "The role of the media in the fight against corruption in Cameroon". I""",
            "Hi today I am going to tell you about my favorite book. My favorite book is called The Hunger Games.",
        ]

        model, params = FlaxGemmaForCausalLM.from_pretrained(model_id, revision="flax", _do_init=False)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="np", padding=True)

        output = model.generate(**inputs, params=params, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

        jit_generate = jax.jit(model.generate)
        output_sequences = jit_generate(**inputs).sequences
        output_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_7b_fp16(self):
        # TODO: change it to the new repo after the release
        model_id = "gg-hf/gemma-7b"
        EXPECTED_TEXTS = [
            """Hello I am doing a project on the topic "The role of the media in the fight against corruption in Cameroon". I""",
            "Hi today I am going to tell you about my favorite book. My favorite book is called The Hunger Games.",
        ]

        model, params = FlaxGemmaForCausalLM.from_pretrained(model_id, revision="flax", _do_init=False, dtype=jnp.float16)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="np", padding=True)

        output = model.generate(**inputs, params=params, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

        jit_generate = jax.jit(model.generate)
        output_sequences = jit_generate(**inputs).sequences
        output_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_7b_bf16(self):
        # TODO: change it to the new repo after the release
        model_id = "gg-hf/gemma-7b"
        EXPECTED_TEXTS = [
            """Hello I am doing a project on the "The effect of the use of a new type of a new type of a""",
            "Hi today I am going to tell you about the new update for the new update is the new update is the",
        ]

        model, params = FlaxGemmaForCausalLM.from_pretrained(model_id, revision="flax", _do_init=False, dtype=jnp.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="np", padding=True)

        output = model.generate(**inputs, params=params, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

        jit_generate = jax.jit(model.generate)
        output_sequences = jit_generate(**inputs).sequences
        output_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)
