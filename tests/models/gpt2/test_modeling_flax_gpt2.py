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

from transformers import GPT2Config, GPT2Tokenizer, is_flax_available
from transformers.testing_utils import require_flax, slow

from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_flax_available():
    import jax
    import jax.numpy as jnp

    from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2LMHeadModel, FlaxGPT2Model


class FlaxGPT2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
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
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = GPT2Config(
            vocab_size=self.vocab_size,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            n_positions=self.max_position_embeddings,
            use_cache=False,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

        return (config, input_ids, input_mask)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict

    def prepare_config_and_inputs_for_decoder(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )

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

    def check_bool_attention_mask_in_generation(self, model_class_name, config, input_ids, attention_mask):
        model = model_class_name(config)

        output_int_att_mask = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
        )

        output_bool_att_mask = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask.astype(bool),
            max_new_tokens=3,
        )

        self.parent.assertTrue(
            (output_bool_att_mask.sequences == output_int_att_mask.sequences).all(),
            "Generated response differ between boolean and integer attention mask",
        )


@require_flax
class FlaxGPT2ModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxGPT2Model, FlaxGPT2LMHeadModel) if is_flax_available() else ()

    def setUp(self):
        self.model_tester = FlaxGPT2ModelTester(self)

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

    def test_bool_attention_mask_in_generation(self):
        for model_class_name in self.all_generative_model_classes:
            config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
            self.model_tester.check_bool_attention_mask_in_generation(
                model_class_name, config, input_ids, attention_mask
            )

    @slow
    def test_batch_generation(self):
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2", pad_token="</s>", padding_side="left")
        inputs = tokenizer(["Hello this is a long string", "Hey"], return_tensors="np", padding=True, truncation=True)

        model = FlaxGPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        model.do_sample = False
        model.config.pad_token_id = model.config.eos_token_id

        jit_generate = jax.jit(model.generate)

        output_sequences = jit_generate(inputs["input_ids"], attention_mask=inputs["attention_mask"]).sequences

        output_string = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        expected_string = [
            "Hello this is a long string of words. I'm going to start with the first one.\n",
            "Hey, I'm not sure if I'm going to be able to do",
        ]

        self.assertListEqual(output_string, expected_string)

    @slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("openai-community/gpt2", from_pt=True)
            outputs = model(np.ones((1, 1)))
            self.assertIsNotNone(outputs)
