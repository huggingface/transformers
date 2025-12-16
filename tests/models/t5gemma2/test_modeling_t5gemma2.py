# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch T5Gemma2 model."""

import copy
import unittest

import pytest

from transformers import (
    T5Gemma2Config,
    T5Gemma2DecoderConfig,
    T5Gemma2EncoderConfig,
    T5Gemma2TextConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin, has_similar_generate_outputs
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import (
        T5Gemma2ForConditionalGeneration,
        T5Gemma2ForSequenceClassification,
        T5Gemma2ForTokenClassification,
        T5Gemma2Model,
    )


class T5Gemma2ModelTester:
    config_class = T5Gemma2Config
    text_config_class = T5Gemma2TextConfig
    encoder_config_class = T5Gemma2EncoderConfig
    decoder_config_class = T5Gemma2DecoderConfig

    if is_torch_available():
        model_class = T5Gemma2Model
        causal_lm_class = T5Gemma2ForConditionalGeneration
        sequence_classification_class = T5Gemma2ForSequenceClassification
        token_classification_class = T5Gemma2ForTokenClassification

    def __init__(
        self,
        parent,
        batch_size=13,
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        vocab_size=99,
        # decoder-specific
        seq_length=7,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        # encoder-specific
        encoder_seq_length=7,
        encoder_hidden_size=32,
        encoder_num_hidden_layers=2,
        encoder_num_attention_heads=4,
        encoder_num_key_value_heads=2,
        encoder_intermediate_size=37,
        # vision-specific
        mm_tokens_per_image=2,
        image_token_index=4,
        boi_token_index=5,
        eoi_token_index=6,
        siglip_config={
            "use_labels": True,
            "image_size": 20,
            "patch_size": 5,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        # common
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        layer_types=["full_attention", "sliding_attention"],
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
        # special ids
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        # decoder
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        # encoder
        self.encoder_seq_length = encoder_seq_length
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_num_key_value_heads = encoder_num_key_value_heads
        self.encoder_intermediate_size = encoder_intermediate_size
        # vision
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_index = image_token_index
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.siglip_config = siglip_config
        self.num_channels = siglip_config["num_channels"]
        self.image_size = siglip_config["image_size"]
        # common
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_types = layer_types
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.head_dim = self.hidden_size // self.num_attention_heads
        # special ids
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def get_encoder_config(self):
        return self.encoder_config_class(
            text_config=self.text_config_class(
                vocab_size=self.vocab_size,
                hidden_size=self.encoder_hidden_size,
                num_hidden_layers=self.encoder_num_hidden_layers,
                num_attention_heads=self.encoder_num_attention_heads,
                num_key_value_heads=self.encoder_num_key_value_heads,
                intermediate_size=self.encoder_intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                layer_types=self.layer_types,
                type_vocab_size=self.type_vocab_size,
                is_decoder=False,
                initializer_range=self.initializer_range,
                head_dim=self.head_dim,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            ),
            # vision.
            vision_config=self.siglip_config,
            image_token_index=self.image_token_index,
            boi_token_index=self.boi_token_index,
            eoi_token_index=self.eoi_token_index,
            mm_tokens_per_image=self.mm_tokens_per_image,
            hidden_size=self.encoder_hidden_size,
        )

    def get_decoder_config(self):
        return self.decoder_config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            cross_attention_hidden_size=self.encoder_hidden_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            layer_types=self.layer_types,
            type_vocab_size=self.type_vocab_size,
            is_decoder=True,
            initializer_range=self.initializer_range,
            head_dim=self.head_dim,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def get_config(self, is_encoder_decoder=True):
        return self.config_class(
            encoder=self.get_encoder_config(),
            decoder=self.get_decoder_config(),
            is_encoder_decoder=is_encoder_decoder,
            # vision.
            image_token_index=self.image_token_index,
            # Used for generation test.
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size - 1) + 1
        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1) + 1
        # Vision inputs.
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.siglip_config["num_channels"],
                self.siglip_config["image_size"],
                self.siglip_config["image_size"],
            ]
        )

        # Remove BOS symbols from inputs.
        input_ids = torch.where(input_ids == self.bos_token_id, 42, input_ids)
        decoder_input_ids = torch.where(decoder_input_ids == self.bos_token_id, 42, decoder_input_ids)

        # Avoid leading PAD tokens from inputs.
        decoder_input_ids[:, 0] = self.pad_token_id + 1

        # set the 3 first tokens to be image, and ensure that no other tokens are image tokens
        # do not change this unless you modified image size or patch size
        input_ids[input_ids == config.encoder.image_token_index] = self.pad_token_id
        input_ids[:, :1] = config.encoder.image_token_index

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        return (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
            pixel_values,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
            pixel_values,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        pixel_values,
    ):
        model = self.model_class(config=config).to(torch_device).eval()

        result = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        decoder_output = result.last_hidden_state
        decoder_past = result.past_key_values
        encoder_output = result.encoder_last_hidden_state

        self.parent.assertEqual(
            encoder_output.size(), (self.batch_size, self.encoder_seq_length, self.encoder_hidden_size)
        )
        self.parent.assertEqual(decoder_output.size(), (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertIsNotNone(decoder_past)
        self.parent.assertEqual(len(decoder_past.self_attention_cache), config.decoder.num_hidden_layers)
        self.parent.assertEqual(len(decoder_past.cross_attention_cache), config.decoder.num_hidden_layers)

    def check_prepare_lm_labels_via_shift_left(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        pixel_values,
    ):
        model = self.model_class(config=config).to(torch_device).eval()

        # _shift_right should be called on labels
        shifted_labels = model.prepare_decoder_input_ids_from_labels(lm_labels)

        # first token should be decoder_start_token_id
        self.parent.assertTrue(torch.all(shifted_labels[:, 0] == config.decoder.bos_token_id))

        # the rest should be the labels shifted by one, with -100 replaced by pad_token_id
        labels_without_ignore_index = lm_labels.masked_fill(lm_labels == -100, config.decoder.pad_token_id)
        self.parent.assertTrue(torch.all(shifted_labels[:, 1:] == labels_without_ignore_index[:, :-1]))

    def create_and_check_with_lm_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        pixel_values,
    ):
        model = self.causal_lm_class(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
            pixel_values=pixel_values,
        )
        self.parent.assertEqual(len(outputs), 4)
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_with_sequence_classification_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        pixel_values,
    ):
        labels = torch.tensor([1] * self.batch_size, dtype=torch.long, device=torch_device)
        model = self.sequence_classification_class(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_with_token_classification_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        pixel_values,
    ):
        labels = torch.tensor([1] * self.seq_length * self.batch_size, dtype=torch.long, device=torch_device)
        model = self.token_classification_class(config=config)
        model = model.to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.seq_length, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_decoder_model_past(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        pixel_values,
    ):
        model = self.model_class(config=config).get_decoder().to(torch_device).eval()
        encoder_hidden_states = torch.ones(
            (self.batch_size, self.encoder_seq_length, self.encoder_hidden_size), dtype=torch.float32
        ).to(torch_device)

        # first forward pass
        outputs = model(decoder_input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=True)
        outputs_use_cache_conf = model(decoder_input_ids, encoder_hidden_states=encoder_hidden_states)
        outputs_no_past = model(decoder_input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)

        output_from_no_past = model(next_input_ids, encoder_hidden_states=encoder_hidden_states)["last_hidden_state"]
        output_from_past = model(
            next_tokens, encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values
        )["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_decoder_model_attention_mask_past(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        pixel_values,
    ):
        model = self.model_class(config=config).get_decoder().to(torch_device).eval()
        encoder_hidden_states = torch.ones(
            (self.batch_size, self.encoder_seq_length, self.encoder_hidden_size), dtype=torch.float32
        ).to(torch_device)

        # create attention mask
        attn_mask = torch.ones(decoder_input_ids.shape, dtype=torch.long, device=torch_device)

        half_seq_length = decoder_input_ids.shape[-1] // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past_key_values = model(
            decoder_input_ids, encoder_hidden_states=encoder_hidden_states, attention_mask=attn_mask, use_cache=True
        ).to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size).squeeze(-1)
        decoder_input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((attn_mask.shape[0], 1), dtype=torch.long, device=torch_device)],
            dim=1,
        )

        # get two different outputs
        output_from_no_past = model(
            next_input_ids, encoder_hidden_states=encoder_hidden_states, attention_mask=attn_mask
        )["last_hidden_state"]
        output_from_past = model(
            next_tokens,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            attention_mask=attn_mask,
        )["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        pixel_values,
    ):
        model = self.model_class(config=config).get_decoder().to(torch_device).eval()
        encoder_hidden_states = torch.ones(
            (self.batch_size, self.encoder_seq_length, self.encoder_hidden_size), dtype=torch.float32
        ).to(torch_device)

        # first forward pass
        outputs = model(
            decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            use_cache=True,
        )

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids, encoder_hidden_states=encoder_hidden_states, attention_mask=next_attention_mask
        )["last_hidden_state"]
        output_from_past = model(
            next_tokens,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
        )["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_generate_with_past_key_values(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        pixel_values,
    ):
        model = self.causal_lm_class(config=config).to(torch_device).eval()
        torch.manual_seed(0)
        output_without_past_cache = model.generate(
            input_ids, pixel_values=pixel_values, num_beams=2, max_length=5, do_sample=True, use_cache=False
        )
        torch.manual_seed(0)
        output_with_past_cache = model.generate(
            input_ids, pixel_values=pixel_values, num_beams=2, max_length=5, do_sample=True
        )
        self.parent.assertTrue(torch.all(output_with_past_cache == output_without_past_cache))

    def create_and_check_model_fp16_forward(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        pixel_values,
    ):
        model = self.model_class(config=config).to(torch_device).half().eval()
        output = model(
            input_ids,
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )["last_hidden_state"]
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_create_and_check_forward_full_mask(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        pixel_values,
    ):
        """
        Checks whether we can use the shortcuts in our mask generation (SDPA) properly,
        these rely on the `is_causal` flag to function properly
        """
        model = self.model_class(config=config).to(torch_device).eval()

        # Force full mask (all true) which can be shortcircuited to `None`
        attention_mask = torch.ones_like(attention_mask)
        decoder_attention_mask = torch.ones_like(decoder_attention_mask)

        output_full_mask = model(
            input_ids,
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )["last_hidden_state"]

        # Compile forces the mask creation to happen at any time
        model.forward = torch.compile(model.forward)
        output_full_mask_no_shortcut = model(
            input_ids,
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )["last_hidden_state"]

        self.parent.assertTrue(torch.allclose(output_full_mask, output_full_mask_no_shortcut, atol=1e-3, rtol=1e-3))


@require_torch
class T5Gemma2ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            T5Gemma2Model,
            T5Gemma2ForConditionalGeneration,
            T5Gemma2ForSequenceClassification,
            T5Gemma2ForTokenClassification,
        )
        if is_torch_available()
        else ()
    )

    _is_stateful = True
    is_encoder_decoder = True

    # MP works but offload doesn't work when the SigLIP MultiheadAttention is offloaded
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def setUp(self):
        self.model_tester = T5Gemma2ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=T5Gemma2Config,
            # For faking the testing.
            hidden_size=37,
            vocab_size=self.model_tester.vocab_size,
            num_attention_heads=self.model_tester.num_attention_heads,
            num_hidden_layers=self.model_tester.num_hidden_layers,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_shift_right(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_prepare_lm_labels_via_shift_left(*config_and_inputs)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # Based on tests.models.t5.test_modeling_t5.T5ModelTest.test_inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (T5Gemma2Model, T5Gemma2ForConditionalGeneration):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_lm_head(*config_and_inputs)

    def test_with_sequence_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_sequence_classification_head(*config_and_inputs)

    def test_with_token_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_token_classification_head(*config_and_inputs)

    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    def test_decoder_model_past_with_attn_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_attention_mask_past(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_generate_with_past_key_values(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generate_with_past_key_values(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Can't do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    def test_forward_full_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_create_and_check_forward_full_mask(*config_and_inputs)

    # Based on tests.models.gemma.test_modeling_gemma.GemmaModelTest.test_Gemma_sequence_classification_model with Gemma -> T5Gemma2 (Add is_encoder_decoder option)
    def test_T5Gemma2_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)

        for pixel_values in [None, input_dict["pixel_values"]]:
            model = self.model_tester.sequence_classification_class(config).to(torch_device).eval()
            result = model(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=sequence_labels)
            self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Based on tests.models.gemma.test_modeling_gemma.GemmaModelTest.test_Gemma_sequence_classification_model_for_single_label with Gemma -> T5Gemma2 (Add is_encoder_decoder option)
    def test_T5Gemma2_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)

        for pixel_values in [None, input_dict["pixel_values"]]:
            model = self.model_tester.sequence_classification_class(config).to(torch_device).eval()
            result = model(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=sequence_labels)
            self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Based on tests.models.gemma.test_modeling_gemma.GemmaModelTest.test_Gemma_sequence_classification_model_for_multi_label with Gemma -> T5Gemma2 (Add is_encoder_decoder option)
    def test_T5Gemma2_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)

        for pixel_values in [None, input_dict["pixel_values"]]:
            model = self.model_tester.sequence_classification_class(config).to(torch_device).eval()
            result = model(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=sequence_labels)
            self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Based on tests.models.gemma.test_modeling_gemma.GemmaModelTest.test_Gemma_token_classification_model with Gemma -> T5Gemma2 (Add is_encoder_decoder option)
    def test_T5Gemma2_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        decoder_input_ids = input_dict["decoder_input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)

        for pixel_values in [None, input_dict["pixel_values"]]:
            model = self.model_tester.token_classification_class(config).to(torch_device).eval()

            result = model(
                input_ids,
                decoder_input_ids=decoder_input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=token_labels,
            )
            self.assertEqual(
                result.logits.shape,
                (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
            )

    @unittest.skip("This was not properly written, submodules need the attribute to be overwritten")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Mismatch issue doesn't exist in T5Gemma2.")
    def test_load_with_mismatched_shapes(self):
        pass

    # Based on tests.generation.test_utils.GenerationTesterMixin.test_generate_continue_from_past_key_values
    # Updated decoder_attention_mask to consider the appended bos token
    @pytest.mark.generate
    def test_generate_continue_from_past_key_values(self):
        # Tests that we can continue generating from past key values, returned from a previous `generate` call
        for model_class in self.all_generative_model_classes:
            if model_class == self.model_tester.token_classification_class:
                continue
            if any(model_name in model_class.__name__.lower() for model_name in ["imagegpt", "mllama"]):
                self.skipTest(reason="Won't fix: old model with unique inputs/caches/other")
            if any(model_name in model_class.__name__.lower() for model_name in ["umt5"]):
                self.skipTest(reason="TODO: needs modeling or test input preparation fixes for compatibility")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            # Let's make it always:
            # 1. use cache (for obvious reasons)
            # 2. generate to max length (which can be achieved by setting the eos token to an invalid value), which
            #    would make the test flaky (e.g. EOS is generated on iteration 1 on both generations, but the
            #    continuation would force it to generate beyond an EOS token)
            # 3. ignore `token_type_ids` for simplicity
            # 4. ignore `forced_eos_token_id`, which requires further manipulation of the continuation inputs and is
            #    active by default on some models
            # 5. ignore `encoder_no_repeat_ngram_size`, which is set by default in some encoder-decoder models. When
            #    we use their decoder as a stand-alone model, `encoder_no_repeat_ngram_size` actually prevents
            #    repetition exclusively from the prompt. This test relies on comparing one call vs 2 calls
            #    with cache, what is considered a prompt is different in the two cases.

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            model = model_class(config).to(torch_device)
            model.eval()

            # If "past_key_values" is not returned, skip the test (e.g. RWKV uses a different cache name and format)
            outputs = model(**inputs)
            if "past_key_values" not in outputs:
                self.skipTest(reason="This model doesn't return `past_key_values`")

            generate_kwargs = {
                "pad_token_id": -1,
                "eos_token_id": -1,
                "forced_eos_token_id": None,
                "encoder_no_repeat_ngram_size": 0,
                "use_cache": True,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values
            outputs = model.generate(**inputs, **generate_kwargs, max_new_tokens=4)

            # Let's generate again, but passing the past key values in between (3 + 1 = 4 tokens). Note that the
            # inputs may need to be tweaked across `generate` calls (like the attention mask).
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=3)

            # Continue from the tokens generated above, preparing the inputs accordingly
            inputs["past_key_values"] = outputs_cached.past_key_values
            new_attention_len = outputs_cached.sequences.shape[-1]

            # It must be encoder-decoder models
            self.assertTrue(config.is_encoder_decoder)

            inputs["decoder_input_ids"] = outputs_cached.sequences
            if "decoder_attention_mask" in inputs:
                decoder_attention_mask = inputs["decoder_attention_mask"]

                # Add BOS mask: the new sequence comes with a new BOS token, which is not included in the original inputs
                padding_tensor = torch.ones_like(decoder_attention_mask[:, :1])
                decoder_attention_mask = torch.cat([padding_tensor, decoder_attention_mask], dim=1)

                inputs["decoder_attention_mask"] = torch.nn.functional.pad(
                    decoder_attention_mask,
                    (0, new_attention_len - decoder_attention_mask.shape[1]),
                    mode="constant",
                    value=1,
                )

            first_caches_scores = outputs_cached.scores
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=1)
            full_cached_scores = first_caches_scores + outputs_cached.scores
            outputs_cached.scores = full_cached_scores

            # The two sets of generated text and past kv should be equal to each other
            self.assertTrue(has_similar_generate_outputs(outputs, outputs_cached))
            self._check_caches_are_equal(outputs.past_key_values, outputs_cached.past_key_values)

    @unittest.skip("T5Gemma 2 only support final layer hidden states.")
    def test_hidden_states_output(self):
        pass

    # Based on tests.models.t5.test_modeling_t5.T5ModelTest.test_custom_4d_attention_mask
    # Excluding the final token from input_ids
    def test_custom_4d_attention_mask(self):
        for model_class in self.all_generative_model_classes:
            config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config).to(device=torch_device, dtype=torch.float32)

            (
                input_ids,
                position_ids,
                input_ids_shared_prefix,
                mask_shared_prefix,
                position_ids_shared_prefix,
            ) = self._get_custom_4d_mask_test_data()
            mask_shared_prefix = mask_shared_prefix == 0.0

            outputs = model.forward(
                decoder_input_ids=input_ids,
                input_ids=input_ids[:, :-1],
                decoder_position_ids=position_ids,
            )
            logits = outputs.logits
            # logits.shape == torch.Size([3, 4, ...])

            outputs_shared_prefix = model(
                input_ids=input_ids[:1, :-1],
                decoder_input_ids=input_ids_shared_prefix,
                decoder_attention_mask=mask_shared_prefix,
                decoder_position_ids=position_ids_shared_prefix,
            )
            logits_shared_prefix = outputs_shared_prefix.logits
            # logits_shared_prefix.shape == torch.Size([1, 6, ...])

            torch.testing.assert_close(
                outputs.encoder_last_hidden_state[0], outputs_shared_prefix.encoder_last_hidden_state[0]
            )

            out_last_tokens = logits[:, -1, :]  # last tokens in each batch line
            out_shared_prefix_last_tokens = logits_shared_prefix[0, -3:, :]  # last three tokens

            # comparing softmax-normalized logits:
            normalized_0 = F.softmax(out_last_tokens)
            normalized_1 = F.softmax(out_shared_prefix_last_tokens)
            torch.testing.assert_close(normalized_0[2], normalized_1[2], rtol=1e-3, atol=1e-4)
            torch.testing.assert_close(normalized_0, normalized_1, rtol=1e-3, atol=1e-4)

    @unittest.skip(reason="SiglipVisionModel (vision backbone) does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SiglipVisionModel (vision backbone) does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SiglipVisionModel (vision backbone) does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="SiglipVisionModel (vision backbone) does not support standalone training")
    def test_torch_compile_for_training(self):
        pass

    @unittest.skip(reason="Self&cross attention are splited after the merged attention")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(
        reason="Merged attention module will always require a mask which is incompatible with the FA backend"
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass
