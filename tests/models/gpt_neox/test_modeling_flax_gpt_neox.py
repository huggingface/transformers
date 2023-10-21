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
""" Testing suite for the Flax GPTNeoX model. """

import unittest

import numpy as np
from parameterized import parameterized

import transformers
from transformers import AutoTokenizer, GPTNeoXConfig, is_flax_available, is_torch_available
from transformers.testing_utils import is_pt_flax_cross_test, require_flax, slow, torch_device

from ...generation.test_flax_utils import FlaxGenerationTesterMixin
from ...test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor


if is_flax_available():
    import jax.numpy as jnp

    from transformers.modeling_flax_pytorch_utils import (
        load_flax_weights_in_pytorch_model,
    )
    from transformers.models.gpt_neox.modeling_flax_gpt_neox import FlaxGPTNeoXForCausalLM, FlaxGPTNeoXModel

if is_torch_available():
    import torch


class FlaxGPTNeoXModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
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
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.pad_token_id = vocab_size - 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = np.tril(np.ones((self.batch_size, self.seq_length)))

        config = GPTNeoXConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=True,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

        return (config, input_ids, input_mask)

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
class FlaxGPTNeoXModelTest(FlaxModelTesterMixin, FlaxGenerationTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxGPTNeoXModel, FlaxGPTNeoXForCausalLM) if is_flax_available() else ()
    all_generative_model_classes = (FlaxGPTNeoXForCausalLM,) if is_flax_available() else ()

    def setUp(self):
        self.model_tester = FlaxGPTNeoXModelTester(self)

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
    def test_batch_generation(self):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer(["Hello this is a long string", "Hey"], return_tensors="np", padding=True, truncation=True)

        model = FlaxGPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped", from_pt=True)
        model.do_sample = False
        model.config.pad_token_id = model.config.eos_token_id

        output_sequences = model.generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"], pad_token_id=tokenizer.pad_token_id
        ).sequences

        output_string = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        expected_string = [
            "Hello this is a long string of characters.\n\nI'm trying to get the first character of",
            "Heyon, I'm not sure if you're aware of this, but",
        ]

        self.assertListEqual(output_string, expected_string)

    @slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("EleutherAI/pythia-410m-deduped", from_pt=True)
            outputs = model(np.ones((1, 1)))
            self.assertIsNotNone(outputs)

    @parameterized.expand([("linear",), ("dynamic",)])
    def test_model_rope_scaling(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        original_max_position_embeddings = config.max_position_embeddings
        scaled_max_position_embeddings = int(config.max_position_embeddings * 1.5)

        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, scaled_max_position_embeddings], config.vocab_size)

        config.max_position_embeddings = scaled_max_position_embeddings
        original_model = FlaxGPTNeoXModel(config, seed=42)
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        config.max_position_embeddings = original_max_position_embeddings
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = FlaxGPTNeoXModel(config, seed=42)
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            self.assertTrue(np.allclose(original_short_output, scaled_short_output, atol=1e-5))
        else:
            self.assertFalse(np.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(np.allclose(original_long_output, scaled_long_output, atol=1e-5))

    @parameterized.expand([("linear",), ("dynamic",)])
    @is_pt_flax_cross_test
    def test_equivalence_rope_scaling(self, scaling_type):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                # Output all for aggressive testing
                config.output_hidden_states = True
                config.output_attentions = self.has_attentions

                # prepare inputs
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                pt_inputs = {k: torch.tensor(v.tolist(), device=torch_device) for k, v in prepared_inputs_dict.items()}

                # load corresponding PyTorch class
                pt_model_class_name = model_class.__name__[4:]  # Skip the "Flax" at the beginning
                pt_model_class = getattr(transformers, pt_model_class_name)

                pt_model = pt_model_class(config).eval()
                # Flax models don't use the `use_cache` option and cache is not returned as a default.
                # So we disable `use_cache` here for PyTorch model.
                pt_model.config.use_cache = False
                fx_model = model_class(config, dtype=jnp.float32)

                pt_model = load_flax_weights_in_pytorch_model(pt_model, fx_model.params)

                # make sure weights are tied in PyTorch
                pt_model.tie_weights()

                # send pytorch model to the correct device
                pt_model.to(torch_device)

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs)
                fx_outputs = fx_model(**prepared_inputs_dict)

                fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs, pt_outputs, model_class)


@require_flax
class FlaxGPTNeoXLanguageGenerationTest(unittest.TestCase):
    @is_pt_flax_cross_test
    def test_lm_generate_gptneox(self):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
        model = FlaxGPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped", from_pt=True)
        inputs = tokenizer(["My favorite food is"], return_tensors="np")

        # The hub repo. is updated on 2023-04-04, resulting in poor outputs.
        # See: https://github.com/huggingface/transformers/pull/24193
        expected_output = "My favorite food is a good old-fashioned, old-fashioned, old-fashioned.\n\nI'm not sure"

        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20).sequences
        output_str = tokenizer.batch_decode(output_ids)[0]

        self.assertEqual(output_str, expected_output)
