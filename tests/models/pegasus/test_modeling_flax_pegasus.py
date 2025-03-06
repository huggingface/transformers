# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

from transformers import PegasusConfig, PegasusTokenizer, is_flax_available
from transformers.testing_utils import require_flax, slow

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
    import numpy as np

    from transformers import FlaxPegasusForConditionalGeneration, FlaxPegasusModel


@require_flax
class FlaxPegasusModelTester:
    config_cls = PegasusConfig
    config_updates = {}
    hidden_act = "gelu"

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
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

        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length - 1], self.vocab_size).clip(3, self.vocab_size)
        eos_tensor = np.expand_dims(np.array([self.eos_token_id] * self.batch_size), 1)
        input_ids = np.concatenate([input_ids, eos_tensor], axis=1)

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.config_cls(
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
            eos_token_ids=[2],
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.pad_token_id,
            **self.config_updates,
        )
        inputs_dict = prepare_pegasus_inputs_dict(config, input_ids, decoder_input_ids)
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


def prepare_pegasus_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
):
    if attention_mask is None:
        attention_mask = np.not_equal(input_ids, config.pad_token_id).astype(np.int8)
    if decoder_attention_mask is None:
        decoder_attention_mask = np.concatenate(
            [
                np.ones(decoder_input_ids[:, :1].shape, dtype=np.int8),
                np.not_equal(decoder_input_ids[:, 1:], config.pad_token_id).astype(np.int8),
            ],
            axis=-1,
        )
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
    }


@require_flax
class FlaxPegasusModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            FlaxPegasusForConditionalGeneration,
            FlaxPegasusModel,
        )
        if is_flax_available()
        else ()
    )
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = FlaxPegasusModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PegasusConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_use_cache_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            self.model_tester.check_use_cache_forward(model_class, config, inputs_dict)

    def test_use_cache_forward_with_attn_mask(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
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
            model = model_class_name.from_pretrained("google/pegasus-large", from_pt=True)
            input_ids = np.ones((1, 1))
            outputs = model(input_ids)
            self.assertIsNotNone(outputs)

    @slow
    def test_pegasus_xsum_summary(self):
        model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

        src_text = [
            """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.""",
            """ The London trio are up for best UK act and best album, as well as getting two nominations in the best song category."We got told like this morning 'Oh I think you're nominated'", said Dappy."And I was like 'Oh yeah, which one?' And now we've got nominated for four awards. I mean, wow!"Bandmate Fazer added: "We thought it's best of us to come down and mingle with everyone and say hello to the cameras. And now we find we've got four nominations."The band have two shots at the best song prize, getting the nod for their Tynchy Stryder collaboration Number One, and single Strong Again.Their album Uncle B will also go up against records by the likes of Beyonce and Kanye West.N-Dubz picked up the best newcomer Mobo in 2007, but female member Tulisa said they wouldn't be too disappointed if they didn't win this time around."At the end of the day we're grateful to be where we are in our careers."If it don't happen then it don't happen - live to fight another day and keep on making albums and hits for the fans."Dappy also revealed they could be performing live several times on the night.The group will be doing Number One and also a possible rendition of the War Child single, I Got Soul.The charity song is a  re-working of The Killers' All These Things That I've Done and is set to feature artists like Chipmunk, Ironik and Pixie Lott.This year's Mobos will be held outside of London for the first time, in Glasgow on 30 September.N-Dubz said they were looking forward to performing for their Scottish fans and boasted about their recent shows north of the border."We just done Edinburgh the other day," said Dappy."We smashed up an N-Dubz show over there. We done Aberdeen about three or four months ago - we smashed up that show over there! Everywhere we go we smash it up!" """,
        ]

        tgt_text = [
            "California's largest electricity provider has turned off power to hundreds of thousands of customers.",
            "Pop group N-Dubz have revealed they were surprised to get four nominations for this year's Mobo Awards.",
        ]

        inputs = tokenizer(src_text, return_tensors="np", truncation=True, max_length=512, padding=True)
        translated_tokens = model.generate(**inputs, num_beams=2).sequences
        decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        assert tgt_text == decoded
