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

import numpy as np

from transformers import AutoTokenizer, GemmaConfig, is_flax_available
from transformers.testing_utils import require_flax, require_read_token, slow

from ...test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor


if is_flax_available():
    import jax
    import jax.numpy as jnp

    from transformers.models.gemma.modeling_flax_gemma import (
        FlaxGemmaForCausalLM,
        FlaxGemmaModel,
    )


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
class FlaxGemmaModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxGemmaModel, FlaxGemmaForCausalLM) if is_flax_available() else ()

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
            model = model_class_name.from_pretrained("google/gemma-2b", from_pt=True)
            outputs = model(np.ones((1, 1)))
            self.assertIsNotNone(outputs)


@slow
@require_flax
@require_read_token
class FlaxGemmaIntegrationTest(unittest.TestCase):
    input_text = ["The capital of France is", "To play the perfect cover drive"]
    model_id = "google/gemma-2b"
    revision = "flax"

    def setUp(self):
        self.model, self.params = FlaxGemmaForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, _do_init=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = "left"

    def test_logits(self):
        inputs = self.tokenizer(self.input_text, return_tensors="np", padding=True)
        # fmt: off
        EXPECTED_MEAN = [
            [-16.427, -21.386, -35.491, -36.258, -31.401, -36.370, -37.598],
            [-21.386, -32.150, -33.155, -34.344, -34.706, -34.678, -38.495],
        ]
        EXPECTED_SLICE = [-33.462, -16.481, -30.837, -32.195, -33.113]
        # fmt: on

        logits = self.model(**inputs, params=self.params).logits

        diff_mean = jnp.abs(logits.mean(-1) - np.array(EXPECTED_MEAN)).max()
        diff_slice = jnp.abs(logits[0, -1, 475:480] - np.array(EXPECTED_SLICE)).max()

        self.assertAlmostEqual(diff_mean, 0, places=3)
        self.assertAlmostEqual(diff_slice, 0, places=3)

    def test_generation(self):
        EXPECTED_TEXTS = [
            "The capital of France is a city of contrasts. It is a city of history, of art, of culture, of fashion",
            "To play the perfect cover drive, you need to have a good technique and a good mindset.\n\nThe cover drive is a shot",
        ]
        inputs = self.tokenizer(self.input_text, return_tensors="np", padding=True)

        output = self.model.generate(**inputs, params=self.params, max_new_tokens=20, do_sample=False)
        output_text = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_jit_generation(self):
        EXPECTED_TEXTS = [
            "The capital of France is a city of contrasts. It is a city of history, culture, and art, but it is",
            "To play the perfect cover drive, you need to have a good technique and a good mindset.\n\nThe cover drive is a shot",
        ]
        inputs = self.tokenizer(self.input_text, return_tensors="np", padding=True)

        def generate(input_ids, attention_mask):
            outputs = self.model.generate(
                input_ids, attention_mask=attention_mask, params=self.params, max_new_tokens=20, do_sample=False
            )
            return outputs

        jit_generate = jax.jit(generate)
        output_sequences = jit_generate(**inputs).sequences
        output_text = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)
