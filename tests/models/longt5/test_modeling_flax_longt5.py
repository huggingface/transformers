# coding=utf-8
# Copyright 2022 Google LongT5 Authors and HuggingFace Inc. team.
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
import tempfile
import unittest

import numpy as np

import transformers
from transformers import is_flax_available
from transformers.models.auto import get_values
from transformers.testing_utils import (
    is_pt_flax_cross_test,
    require_flax,
    require_sentencepiece,
    require_tokenizers,
    slow,
)

from ...generation.test_flax_utils import FlaxGenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor


if is_flax_available():
    import os

    # The slow tests are often failing with OOM error on GPU
    # This makes JAX allocate exactly what is needed on demand, and deallocate memory that is no longer needed
    # but will be slower as stated here https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax
    import jax.numpy as jnp
    from flax.core.frozen_dict import unfreeze
    from flax.traverse_util import flatten_dict

    from transformers import FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING, FLAX_MODEL_MAPPING, AutoTokenizer, LongT5Config
    from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model
    from transformers.models.longt5.modeling_flax_longt5 import (
        FlaxLongT5ForConditionalGeneration,
        FlaxLongT5Model,
        shift_tokens_right,
    )


class FlaxLongT5ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=9,
        local_radius=5,
        encoder_attention_type="local",
        global_block_size=3,
        # For common tests
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        decoder_start_token_id=0,
        scope=None,
        decoder_layers=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.local_radius = local_radius
        self.block_len = local_radius + 1
        self.encoder_attention_type = encoder_attention_type
        self.global_block_size = global_block_size
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.scope = None
        self.decoder_layers = decoder_layers

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

        config = LongT5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            local_radius=self.local_radius,
            encoder_attention_type=self.encoder_attention_type,
            global_block_size=self.global_block_size,
        )

        return (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
    ):
        model = FlaxLongT5Model(config=config)
        result = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        result = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        decoder_output = result.last_hidden_state
        encoder_output = result.encoder_last_hidden_state

        self.parent.assertEqual(encoder_output.shape, (self.batch_size, self.encoder_seq_length, self.hidden_size))
        self.parent.assertEqual(decoder_output.shape, (self.batch_size, self.decoder_seq_length, self.hidden_size))

    def check_use_cache_forward_with_attn_mask(
        self,
        model_class_name,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
    ):
        max_decoder_length = 20
        model = model_class_name(config)

        encoder_outputs = model.encode(input_ids)

        # prevent fully zero'd out attention mask
        decoder_attention_mask = jnp.ones_like(decoder_attention_mask)

        decoder_attention_mask_cache = jnp.concatenate(
            [
                decoder_attention_mask,
                jnp.zeros((decoder_attention_mask.shape[0], max_decoder_length - decoder_attention_mask.shape[1])),
            ],
            axis=-1,
        )

        past_key_values = model.init_cache(decoder_input_ids.shape[0], max_decoder_length, encoder_outputs)

        outputs_cache = model.decode(
            decoder_input_ids[:, :-1],
            encoder_outputs,
            decoder_attention_mask=decoder_attention_mask_cache,
            past_key_values=past_key_values,
        )
        outputs_cache_next = model.decode(
            decoder_input_ids[:, -1:],
            encoder_outputs,
            past_key_values=outputs_cache.past_key_values,
            decoder_attention_mask=decoder_attention_mask_cache,
        )

        outputs = model.decode(decoder_input_ids, encoder_outputs, decoder_attention_mask=decoder_attention_mask)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict


@require_flax
class FlaxLongT5ModelTest(FlaxModelTesterMixin, FlaxGenerationTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxLongT5Model, FlaxLongT5ForConditionalGeneration) if is_flax_available() else ()
    all_generative_model_classes = (FlaxLongT5ForConditionalGeneration,) if is_flax_available() else ()
    is_encoder_decoder = True

    def setUp(self):
        self.model_tester = FlaxLongT5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LongT5Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_v1_1(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        # check that gated gelu feed forward and different word embeddings work
        config = config_and_inputs[0]
        config.tie_word_embeddings = False
        config.feed_forward_proj = "gated-gelu"
        self.model_tester.create_and_check_model(config, *config_and_inputs[1:])

    def test_use_cache_forward_with_attn_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            self.model_tester.check_use_cache_forward_with_attn_mask(model_class, *config_and_inputs)

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

    def test_shift_right(self):
        decoder_start_token_id = 0
        pad_token_id = 1
        labels = np.arange(2, 102).reshape(5, 20)
        labels[:2, 15:] = -100

        decoder_input_ids = shift_tokens_right(labels, pad_token_id, decoder_start_token_id)
        np_decoder_input_ids = np.array(decoder_input_ids)

        padded_slice = np_decoder_input_ids[:2, (15 + 1) :]
        self.assertTrue((padded_slice == 1).all())

        not_padded_slice = np_decoder_input_ids[2:, 1:]
        rolled_labels = np.roll(labels[2:], 1)[:, 1:]
        self.assertTrue((not_padded_slice == rolled_labels).all())
        self.assertTrue((np_decoder_input_ids[:, 0] == 0).all())

    # overwrite since special base model prefix is used
    def test_save_load_from_base(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = FLAX_MODEL_MAPPING[config.__class__]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            model = base_class(config)
            base_params = flatten_dict(unfreeze(model.params))

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                head_model = model_class.from_pretrained(tmpdirname)

                base_param_from_head = flatten_dict(unfreeze(head_model.params))

                for key in base_param_from_head.keys():
                    max_diff = (base_params[key] - base_param_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    # overwrite since special base model prefix is used
    def test_save_load_to_base(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = FLAX_MODEL_MAPPING[config.__class__]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            model = model_class(config)
            base_params_from_head = flatten_dict(unfreeze(model.params))

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                base_model = base_class.from_pretrained(tmpdirname)

                base_params = flatten_dict(unfreeze(base_model.params))

                for key in base_params_from_head.keys():
                    max_diff = (base_params[key] - base_params_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_length = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_length)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_length)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        block_len = getattr(self.model_tester, "block_len", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, block_len, 3 * block_len],
            )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # Question Answering model returns start_logits and end_logits
                if model_class in get_values(FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                    correct_outlen += 1  # start_logits and end_logits instead of only 1 output

                self.assertEqual(out_len, correct_outlen)

                # decoder attentions
                decoder_attentions = outputs.decoder_attentions
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

                # cross attentions
                cross_attentions = outputs.cross_attentions
                self.assertIsInstance(cross_attentions, (list, tuple))
                self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        encoder_key_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, block_len, 3 * block_len],
            )

    # overwrite since special base model prefix is used
    @is_pt_flax_cross_test
    def test_save_load_from_base_pt(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = FLAX_MODEL_MAPPING[config.__class__]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            model = base_class(config)
            base_params = flatten_dict(unfreeze(model.params))

            # convert Flax model to PyTorch model
            pt_model_class = getattr(transformers, base_class.__name__[4:])  # Skip the "Flax" at the beginning
            pt_model = pt_model_class(config).eval()
            pt_model = load_flax_weights_in_pytorch_model(pt_model, model.params)

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                # save pt model
                pt_model.save_pretrained(tmpdirname)
                head_model = model_class.from_pretrained(tmpdirname, from_pt=True)

                base_param_from_head = flatten_dict(unfreeze(head_model.params))

                for key in base_param_from_head.keys():
                    max_diff = (base_params[key] - base_param_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    # overwrite since special base model prefix is used
    @is_pt_flax_cross_test
    def test_save_load_to_base_pt(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = FLAX_MODEL_MAPPING[config.__class__]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            model = model_class(config)
            base_params_from_head = flatten_dict(unfreeze(model.params))

            # convert Flax model to PyTorch model
            pt_model_class = getattr(transformers, model_class.__name__[4:])  # Skip the "Flax" at the beginning
            pt_model = pt_model_class(config).eval()
            pt_model = load_flax_weights_in_pytorch_model(pt_model, model.params)

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_model.save_pretrained(tmpdirname)
                base_model = base_class.from_pretrained(tmpdirname, from_pt=True)

                base_params = flatten_dict(unfreeze(base_model.params))

                for key in base_params_from_head.keys():
                    max_diff = (base_params[key] - base_params_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    # overwrite since special base model prefix is used
    @is_pt_flax_cross_test
    def test_save_load_bf16_to_base_pt(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = FLAX_MODEL_MAPPING[config.__class__]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            model = model_class(config)
            model.params = model.to_bf16(model.params)
            base_params_from_head = flatten_dict(unfreeze(model.params))

            # convert Flax model to PyTorch model
            pt_model_class = getattr(transformers, model_class.__name__[4:])  # Skip the "Flax" at the beginning
            pt_model = pt_model_class(config).eval()
            pt_model = load_flax_weights_in_pytorch_model(pt_model, model.params)

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_model.save_pretrained(tmpdirname)
                base_model = base_class.from_pretrained(tmpdirname, from_pt=True)

                base_params = flatten_dict(unfreeze(base_model.params))

                for key in base_params_from_head.keys():
                    max_diff = (base_params[key] - base_params_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")


class FlaxLongT5TGlobalModelTest(FlaxLongT5ModelTest):
    def setUp(self):
        self.model_tester = FlaxLongT5ModelTester(self, encoder_attention_type="transient-global")
        self.config_tester = ConfigTester(self, config_class=LongT5Config, d_model=37)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_length = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_length)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_length)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        block_len = getattr(self.model_tester, "block_len", None)
        global_block_size = getattr(self.model_tester, "global_block_size", None)
        global_seq_len = encoder_seq_length // global_block_size

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, block_len, 3 * block_len + global_seq_len],
            )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # Question Answering model returns start_logits and end_logits
                if model_class in get_values(FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                    correct_outlen += 1  # start_logits and end_logits instead of only 1 output

                self.assertEqual(out_len, correct_outlen)

                # decoder attentions
                decoder_attentions = outputs.decoder_attentions
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

                # cross attentions
                cross_attentions = outputs.cross_attentions
                self.assertIsInstance(cross_attentions, (list, tuple))
                self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        encoder_key_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, block_len, 3 * block_len + global_seq_len],
            )


@require_sentencepiece
@require_tokenizers
@require_flax
class FlaxLongT5ModelIntegrationTests(unittest.TestCase):
    model_path = "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"

    def expected_summary(self):
        return [
            "background : coronary artery disease ( cad ) is the emerging cause of morbidity and mortality in"
            " developing world . it provides an excellent resolution for visualization of the coronary arteries for"
            " catheter - based or operating interventions . although the association of this technique with major"
            " complications such as mortality is highly uncommon , it is frequently associated with various cardiac"
            " and noncardiac complications . computed tomography coronary angiography is a promising technique for the"
            " evaluation of cad noninvasively . it assesses disease within the coronary artery and provides"
            " qualitative and quantitative information about nonobstructive atherosclerotic plaque"
        ]

    @slow
    def test_summarization(self):
        model = FlaxLongT5ForConditionalGeneration.from_pretrained(self.model_path)
        tok = AutoTokenizer.from_pretrained(self.model_path)

        ARTICLE = """coronary artery disease ( cad ) is the emerging cause of morbidity and mortality in developing world . \n it provides an excellent resolution for visualization of the coronary arteries for catheter - based or operating interventions . \n
            although the association of this technique with major complications such as mortality is highly uncommon , it is frequently associated with various cardiac and noncardiac complications . computed tomography ( ct ) coronary angiography is
            a promising technique for the evaluation of cad noninvasively . \n it assesses disease within the coronary artery and provides qualitative and quantitative information about nonobstructive atherosclerotic plaque burden within the vessel
            wall . \n thus , ct angiography - based disease evaluation may provide clinically more significant information than conventional angiography . the introduction of multi - slice computed tomography ( msct ) technology such as 64-slice , 12
            8-slice , 256-slice , and now 320-slice msct has produced a high diagnostic accuracy of ct coronary angiography . \n it has consistently showed to have a very high negative predictive value ( well above 90% ) in ruling out patients with s
            ignificant cad defined as coronary luminal stenosis of > 50% . \n the american college of cardiology / american heart association recommends that coronary angiography should be performed before valve surgery in men aged > 40 years , women
            aged > 35 years with coronary risk factors and in postmenopausal women . \n the prevalence of cad in patients undergoing valve replacement is 2040% in developed countries . in the previous studies , \n the incidence of angiographically p
            roven cad in acquired valvular diseases has been shown to vary widely from 9% to 41% . in aortic stenosis , \n we aimed to report the diagnostic performance of 128-slice ct coronary angiography in 50 patients undergoing for major noncoron
            ary cardiac surgery referred for diagnostic invasive coronary angiography to assess the extent and severity of coronary stenosis . \n during january 2013 to december 2014 , we enrolled fifty major noncoronary cardiac surgery patients sche
            duled for invasive coronary angiography who fulfilled the following inclusion criteria of age 40 years , having low or intermediate probability of cad , left ventricular ejection fraction ( lvef ) > 35% , and patient giving informed conse
            nt for undergoing msct and conventional coronary angiography . \n those having any contraindication for contrast injection , lvef < 35% , high pretest probability of cad , and hemodynamic instability were excluded from the study . \n pati
            ents with heart rates of > 70 bpm received ( unless they had known overt heart failure or electrocardiogram ( ecg ) atrioventricular conduction abnormalities ) a single oral dose of 100 mg metoprolol 45 min before the scan . \n patients w
            ith heart rates of > 80 bpm received an additional oral dose of metoprolol if not contraindicated . \n all patients were scanned with a 128-slice ct scanner ( siemens , somatom definition as ) equipped with a new feature in msct technolog
            y , so - called z - axis flying - focus technology . \n the central 32 detector rows acquire 0.6-mm slices , and the flying - focus spot switches back and forth between 2 z positions between each reading . \n two slices per detector row a
            re acquired , which results in a higher oversampling rate in the z - axis , thereby reducing artifacts related to the spiral acquisition and improving spatial resolution down to 0.4 mm . \n a bolus of 6580 ml contrast material ( omnipaque
            ) was injected through an arm vein at a flow rate of 5 ml / s . \n a bolus tracking technique was used to synchronize the arrival of contrast in the coronary arteries with the initiation of the scan . to monitor the arrival of contrast m
            aterial , \n axial scans were obtained at the level of the ascending aorta with a delay of 10 s after the start of the contrast injection . \n the scan was automatically started when a threshold of 150 hounsfield units was reached in a re
            gion of interest positioned in the ascending aorta . \n images were reconstructed with ecg gating to obtain optimal , motion - free image quality . \n all scans were performed within 2 weeks of the msct coronary diagnostic angiogram . a s
            ingle observer unaware of the multi - slice ct results identified coronary lesion as a single vessel , double vessel , or triple vessel disease . \n all lesion , regardless of size , were included for comparison with ct coronary angiograp
            hy . \n lesions were classified as having nonsignificant disease ( luminal irregularities or < 50% stenosis ) or as having significant stenosis . \n stenosis was evaluated in two orthogonal views and classified as significant if the mean
            lumen diameter reduction was 50% using a validated quantitative coronary angiography ( qca ) . \n all scans were analyzed independently by a radiologist and a cardiologist who were unaware of the results of conventional coronary angiograp
            hy . \n total calcium scores of all patients were calculated with dedicated software and expressed as agatston scores . \n the agatston score is a commonly used scoring method that calculates the total amount of calcium on the basis of th
            e number , areas , and peak hounsfield units of the detected calcified lesions . \n all available coronary segments were visually scored for the presence of > 50% considered as significant stenosis . \n maximum intensity projections were
            used to identify coronary lesions and ( curved ) multiplanar reconstructions to classify lesions as significant or nonsignificant . \n data were analyzed using statistical system spss version 20 software ( chicago , il , usa ) . \n the di
            agnostic performance of ct coronary angiography for the detection of significant lesions in coronary arteries with qca as the standard of reference is presented as sensitivity , specificity , positive and negative predictive values , and
            positive and negative likelihood ratios with the corresponding exact 95% of confidence interval ( cis ) . \n comparison between ct and conventional coronary angiography was performed on the two level vessel by vessel ( no or any disease p
            er vessel ) , and patient by patient ( no or any disease per patient ) . \n all scans were performed within 2 weeks of the msct coronary diagnostic angiogram . a single observer unaware of the multi - slice ct results identified coronary
            lesion as a single vessel , double vessel , or triple vessel disease . \n all lesion , regardless of size , were included for comparison with ct coronary angiography . \n lesions were classified as having nonsignificant disease ( luminal
            irregularities or < 50% stenosis ) or as having significant stenosis . \n stenosis was evaluated in two orthogonal views and classified as significant if the mean lumen diameter reduction was 50% using a validated quantitative coronary an
            giography ( qca ) . \n all scans were analyzed independently by a radiologist and a cardiologist who were unaware of the results of conventional coronary angiography . \n total calcium scores of all patients were calculated with dedicated
            software and expressed as agatston scores . \n the agatston score is a commonly used scoring method that calculates the total amount of calcium on the basis of the number , areas , and peak hounsfield units of the detected calcified lesi
            ons . \n all available coronary segments were visually scored for the presence of > 50% considered as significant stenosis . \n maximum intensity projections were used to identify coronary lesions and ( curved ) multiplanar reconstruction
            s to classify lesions as significant or nonsignificant . \n data were analyzed using statistical system spss version 20 software ( chicago , il , usa ) . \n the diagnostic performance of ct coronary angiography for the detection of signif
            icant lesions in coronary arteries with qca as the standard of reference is presented as sensitivity , specificity , positive and negative predictive values , and positive and negative likelihood ratios with the corresponding exact 95% of
            confidence interval ( cis ) . \n comparison between ct and conventional coronary angiography was performed on the two level vessel by vessel ( no or any disease per vessel ) , and patient by patient ( no or any disease per patient ) . \n
            in this study , 29 ( 58% ) subjects were female , and 21 ( 42% ) were male showing an average age of 50.36  8.39 years . \n of fifty patients 24 ( 48% ) , 13 ( 26% ) , eight ( 16% ) , and five ( 10% ) underwent mitral valve replacement ,
            double valve replacement ( dvr ) , aortic valve replacement , and other surgeries , respectively . \n high distribution of cad risk factors such as hypertension ( 24% ) , smoking ( 22% ) , and dyslipidemia ( 18% ) was observed in the stu
            dy group . \n the mean creatinine level was 0.766  0.17 and average dye used in conventional angiography was 48.5  26.6 whereas for ct angiography it was 72.8  6.32 . \n average radiation dose in conventional coronary angiography and msct
            coronary angiography was 5.2 msv and 9.2 msv , respectively . \n the majority of the patients had sinus rhythm ( 68% ) , whereas atrial fibrillation was found in 32% of the subjects . \n patients included in the study had low to intermed
            iate probability of cad . in this study , three patients had complications after conventional angiography . \n complications were of local site hematoma , acute kidney injury managed conservatively , and acute heart failure . \n a patient
            who developed hematoma was obese female patients with body mass index > 30 kg / m . \n the patient suffered from pseudoaneurysm , had hospitalized for 9 days , which leads to increased morbidity and cost of hospital stay . \n the diagnos
            tic accuracy of ct coronary angiography was evaluated regarding true positive , true negative values and is presented in table 1 . the overall sensitivity and \n specificity of ct angiography technique was 100% ( 95% ci : 39.76%100% ) and
            91.30% ( 95% ci : 79.21%97.58% ) , respectively [ table 2 ] . \n the positive predictive value ( 50% ; 95% ci : 15.70%84.30% ) and negative predictive value ( 100% ; 95% ci : 91.59%100% ) of ct angiography were also fairly high in these
            patients . \n recent reports from multiple studies demonstrated that recent - generation msct scanners showed promise for noninvasive detection of coronary stenosis however , until now no studies were found regarding the clinical efficacy
            or prognostic value of 128-slice ct coronary angiography versus conventional invasive coronary angiography in the diagnosis of patients planned for major noncoronary surgeries such as dvr , bentall , atrial septal defect closure , etc .
            in our study , we reported 8% cad prevalence in patients planned for major noncoronary cardiac surgery . \n we performed conventional and msct coronary angiography in all patients and the results showed that ct coronary angiography with i
            nvasive coronary angiography as the reference standard had a considerably high sensitivity ( 100% ) and specificity ( 95.65% ) . \n the health economic model using invasive coronary angiography as the reference standard showed that at a p
            retest probability of cad of 70% or lower , ct coronary angiography resulted in lower cost per patient with a true positive diagnosis . at a pretest probability of cad of 70% or higher , invasive coronary angiography was associated with a
            lower cost per patient with a true positive diagnosis . in our study population , \n two patients developed local site complications in the form of hematoma and pseudoaneurysm after conventional angiography . \n hence , msct coronary ang
            iography will be more favorable in female obese patients with intermediate likelihood of cad . \n hence , msct coronary angiography will be cost - effective in patients of valvular heart diseases . \n however , ct angiography suffers from
            a drawback that average amount of dye used in msct coronary angiography were 72.8  6.32 ml which is higher than average amount of dye required for conventional angiography ( 48.6  26.6 ml ) . \n hence , the use of ct coronary angiography
            could not be used in patients with known renal dysfunction , where reduction of contrast dye load is highly advocated . \n our results show that 128-slice ct coronary angiography is a reliable technique to detect coronary stenosis in pat
            ients planned for noncoronary cardiac surgery . \n although there has been important technological progress in the development of ct coronary angiography , its clinical application remains limited . \n a study wth large numbers of patient
            s is required for the recommendation of only ct coronary angiography for the coronary evaluation in major non - cardiac surgeries . \n mehta institute of cardiology and research center ( affiliated to bj medical college , ahmedabad , guja
            rat , india ) . \n u.n . mehta institute of cardiology and research center ( affiliated to bj medical college , ahmedabad , gujarat , india ) . \n """

        dct = tok(
            [ARTICLE],
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        hypotheses_batch = model.generate(
            **dct,
            num_beams=4,
            length_penalty=2.0,
            max_length=142,
            min_length=56,
            do_sample=False,
            early_stopping=True,
        ).sequences

        decoded = tok.batch_decode(hypotheses_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertListEqual(
            self.expected_summary(),
            decoded,
        )
