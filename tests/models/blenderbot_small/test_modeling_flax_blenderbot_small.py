# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import timeout_decorator  # noqa

from transformers import BlenderbotSmallConfig, is_flax_available
from transformers.testing_utils import require_flax, slow

from ...generation.test_flax_utils import FlaxGenerationTesterMixin
from ...test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor


if is_flax_available():
    import os

    # The slow tests are often failing with OOM error on GPU
    # This makes JAX allocate exactly what is needed on demand, and deallocate memory that is no longer needed
    # but will be slower as stated here https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax
    import jax.numpy as jnp

    from transformers.models.blenderbot_small.modeling_flax_blenderbot_small import (
        FlaxBlenderbotSmallForConditionalGeneration,
        FlaxBlenderbotSmallModel,
        shift_tokens_right,
    )


def prepare_blenderbot_inputs_dict(
    config,
    input_ids,
    decoder_input_ids=None,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = np.where(input_ids != config.pad_token_id, 1, 0)
    if decoder_attention_mask is None:
        decoder_attention_mask = np.where(decoder_input_ids != config.pad_token_id, 1, 0)
    if head_mask is None:
        head_mask = np.ones((config.encoder_layers, config.encoder_attention_heads))
    if decoder_head_mask is None:
        decoder_head_mask = np.ones((config.decoder_layers, config.decoder_attention_heads))
    if cross_attn_head_mask is None:
        cross_attn_head_mask = np.ones((config.decoder_layers, config.decoder_attention_heads))
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
    }


class FlaxBlenderbotSmallModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=50,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        initializer_range=0.02,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.initializer_range = initializer_range

    def prepare_config_and_inputs(self):
        input_ids = np.clip(ids_tensor([self.batch_size, self.seq_length - 1], self.vocab_size), 3, self.vocab_size)
        input_ids = np.concatenate((input_ids, 2 * np.ones((self.batch_size, 1), dtype=np.int64)), -1)

        decoder_input_ids = shift_tokens_right(input_ids, 1, 2)

        config = BlenderbotSmallConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            initializer_range=self.initializer_range,
            use_cache=False,
        )
        inputs_dict = prepare_blenderbot_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def check_use_cache_forward(self, model_class_name, config, inputs_dict):
        max_decoder_length = 20
        model = model_class_name(config)

        encoder_outputs = model.encode(inputs_dict["input_ids"])

        decoder_input_ids, decoder_attention_mask = (
            inputs_dict["decoder_input_ids"],
            inputs_dict["decoder_attention_mask"],
        )

        past_key_values = model.init_cache(decoder_input_ids.shape[0], max_decoder_length, encoder_outputs)
        decoder_attention_mask = jnp.ones((decoder_input_ids.shape[0], max_decoder_length), dtype="i4")

        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(decoder_input_ids.shape[-1] - 1)[None, :],
            (decoder_input_ids.shape[0], decoder_input_ids.shape[-1] - 1),
        )
        outputs_cache = model.decode(
            decoder_input_ids[:, :-1],
            encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            decoder_position_ids=decoder_position_ids,
        )

        decoder_position_ids = jnp.array(decoder_input_ids.shape[0] * [[decoder_input_ids.shape[-1] - 1]], dtype="i4")
        outputs_cache_next = model.decode(
            decoder_input_ids[:, -1:],
            encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=outputs_cache.past_key_values,
            decoder_position_ids=decoder_position_ids,
        )

        outputs = model.decode(decoder_input_ids, encoder_outputs)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")

    def check_use_cache_forward_with_attn_mask(self, model_class_name, config, inputs_dict):
        max_decoder_length = 20
        model = model_class_name(config)

        encoder_outputs = model.encode(inputs_dict["input_ids"])

        decoder_input_ids, decoder_attention_mask = (
            inputs_dict["decoder_input_ids"],
            inputs_dict["decoder_attention_mask"],
        )

        decoder_attention_mask_cache = jnp.concatenate(
            [
                decoder_attention_mask,
                jnp.zeros((decoder_attention_mask.shape[0], max_decoder_length - decoder_attention_mask.shape[1])),
            ],
            axis=-1,
        )

        past_key_values = model.init_cache(decoder_input_ids.shape[0], max_decoder_length, encoder_outputs)
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(decoder_input_ids.shape[-1] - 1)[None, :],
            (decoder_input_ids.shape[0], decoder_input_ids.shape[-1] - 1),
        )

        outputs_cache = model.decode(
            decoder_input_ids[:, :-1],
            encoder_outputs,
            decoder_attention_mask=decoder_attention_mask_cache,
            past_key_values=past_key_values,
            decoder_position_ids=decoder_position_ids,
        )
        decoder_position_ids = jnp.array(decoder_input_ids.shape[0] * [[decoder_input_ids.shape[-1] - 1]], dtype="i4")
        outputs_cache_next = model.decode(
            decoder_input_ids[:, -1:],
            encoder_outputs,
            past_key_values=outputs_cache.past_key_values,
            decoder_attention_mask=decoder_attention_mask_cache,
            decoder_position_ids=decoder_position_ids,
        )

        outputs = model.decode(decoder_input_ids, encoder_outputs, decoder_attention_mask=decoder_attention_mask)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")


@require_flax
class BlenderbotHeadTests(unittest.TestCase):
    vocab_size = 99

    def _get_config_and_data(self):
        input_ids = np.array(
            [
                [71, 82, 18, 33, 46, 91, 2],
                [68, 34, 26, 58, 30, 82, 2],
                [5, 97, 17, 39, 94, 40, 2],
                [76, 83, 94, 25, 70, 78, 2],
                [87, 59, 41, 35, 48, 66, 2],
                [55, 13, 16, 58, 5, 2, 1],  # note padding
                [64, 27, 31, 51, 12, 75, 2],
                [52, 64, 86, 17, 83, 39, 2],
                [48, 61, 9, 24, 71, 82, 2],
                [26, 1, 60, 48, 22, 13, 2],
                [21, 5, 62, 28, 14, 76, 2],
                [45, 98, 37, 86, 59, 48, 2],
                [70, 70, 50, 9, 28, 0, 2],
            ],
            dtype=np.int64,
        )

        batch_size = input_ids.shape[0]
        config = BlenderbotSmallConfig(
            vocab_size=self.vocab_size,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
        )
        return config, input_ids, batch_size

    # @timeout_decorator.timeout(1)  # not working with the decorator so far
    def test_lm_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        lm_model = FlaxBlenderbotSmallForConditionalGeneration(config)
        outputs = lm_model(input_ids=input_ids)
        expected_shape = (batch_size, input_ids.shape[1], config.vocab_size)
        self.assertEqual(outputs["logits"].shape, expected_shape)

    def test_lm_uneven_forward(self):
        config = BlenderbotSmallConfig(
            vocab_size=self.vocab_size,
            d_model=14,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=8,
            decoder_ffn_dim=8,
            max_position_embeddings=48,
        )
        lm_model = FlaxBlenderbotSmallForConditionalGeneration(config)
        context = np.array([[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 2, 1]], dtype=np.int64)
        summary = np.array([[82, 71, 82, 18, 2], [58, 68, 2, 1, 1]], dtype=np.int64)
        outputs = lm_model(input_ids=context, decoder_input_ids=summary)
        expected_shape = (*summary.shape, config.vocab_size)
        self.assertEqual(outputs["logits"].shape, expected_shape)

    def test_shift_tokens_right(self):
        input_ids = np.array([[71, 82, 18, 33, 2, 1, 1], [68, 34, 26, 58, 30, 82, 2]], dtype=np.int64)
        shifted = shift_tokens_right(input_ids, 1, 2)
        n_pad_before = np.equal(input_ids, 1).astype(np.float32).sum()
        n_pad_after = np.equal(shifted, 1).astype(np.float32).sum()
        self.assertEqual(shifted.shape, input_ids.shape)
        self.assertEqual(n_pad_after, n_pad_before - 1)
        self.assertTrue(np.equal(shifted[:, 0], 2).all())


@require_flax
class FlaxBlenderbotSmallModelTest(FlaxModelTesterMixin, unittest.TestCase, FlaxGenerationTesterMixin):
    is_encoder_decoder = True
    all_model_classes = (
        (
            FlaxBlenderbotSmallModel,
            FlaxBlenderbotSmallForConditionalGeneration,
        )
        if is_flax_available()
        else ()
    )
    all_generative_model_classes = (FlaxBlenderbotSmallForConditionalGeneration,) if is_flax_available() else ()

    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        return pipeline_test_casse_name == "TextGenerationPipelineTests"

    def setUp(self):
        self.model_tester = FlaxBlenderbotSmallModelTester(self)

    def test_use_cache_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            self.model_tester.check_use_cache_forward(model_class, config, inputs_dict)

    def test_use_cache_forward_with_attn_mask(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            self.model_tester.check_use_cache_forward_with_attn_mask(model_class, config, inputs_dict)

    def test_encode(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                model = model_class(config)

                @jax.jit
                def encode_jitted(input_ids, attention_mask=None, **kwargs):
                    return model.encode(input_ids=input_ids, attention_mask=attention_mask)

                with self.subTest("JIT Enabled"):
                    jitted_outputs = encode_jitted(**prepared_inputs_dict).to_tuple()

                with self.subTest("JIT Disabled"):
                    with jax.disable_jit():
                        outputs = encode_jitted(**prepared_inputs_dict).to_tuple()

                self.assertEqual(len(outputs), len(jitted_outputs))
                for jitted_output, output in zip(jitted_outputs, outputs):
                    self.assertEqual(jitted_output.shape, output.shape)

    def test_decode(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                model = model_class(config)
                encoder_outputs = model.encode(inputs_dict["input_ids"], inputs_dict["attention_mask"])

                prepared_inputs_dict = {
                    "decoder_input_ids": inputs_dict["decoder_input_ids"],
                    "decoder_attention_mask": inputs_dict["decoder_attention_mask"],
                    "encoder_outputs": encoder_outputs,
                }

                @jax.jit
                def decode_jitted(decoder_input_ids, decoder_attention_mask, encoder_outputs):
                    return model.decode(
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        encoder_outputs=encoder_outputs,
                    )

                with self.subTest("JIT Enabled"):
                    jitted_outputs = decode_jitted(**prepared_inputs_dict).to_tuple()

                with self.subTest("JIT Disabled"):
                    with jax.disable_jit():
                        outputs = decode_jitted(**prepared_inputs_dict).to_tuple()

                self.assertEqual(len(outputs), len(jitted_outputs))
                for jitted_output, output in zip(jitted_outputs, outputs):
                    self.assertEqual(jitted_output.shape, output.shape)

    @slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("facebook/blenderbot_small-90M")
            # FlaxBlenderbotForSequenceClassification expects eos token in input_ids
            input_ids = np.ones((1, 1)) * model.config.eos_token_id
            outputs = model(input_ids)
            self.assertIsNotNone(outputs)
