# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch SeamlessM4T model. """


import copy
import tempfile
import unittest

from transformers import SeamlessM4TConfig, is_speech_available, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.trainer_utils import set_seed
from transformers.utils import cached_property

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        SeamlessM4TForSpeechToSpeech,
        SeamlessM4TForSpeechToText,
        SeamlessM4TForTextToSpeech,
        SeamlessM4TForTextToText,
        SeamlessM4TModel,
    )
    from transformers.models.seamless_m4t.modeling_seamless_m4t import (
        SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

if is_speech_available():
    from transformers import SeamlessM4TProcessor


class SeamlessM4TModelTester:
    def __init__(
        self,
        parent,
        input_modality="speech",
        batch_size=2,
        seq_length=4,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        max_new_tokens=None,
        num_labels=3,
        num_choices=4,
        scope=None,
        vocab_size=20,
        t2u_vocab_size=20,
        hidden_size=6,
        num_hidden_layers=2,
        intermediate_size=6,
        max_position_embeddings=256,
        encoder_layers=2,
        decoder_layers=2,
        encoder_ffn_dim=6,
        decoder_ffn_dim=6,
        t2u_encoder_layers=2,
        t2u_decoder_layers=2,
        t2u_encoder_ffn_dim=6,
        t2u_decoder_ffn_dim=6,
        num_heads=2,
        vocoder_num_spkrs=5,
        vocoder_num_langs=5,
        upsample_initial_channel=32,
        unit_embed_dim=25,
        spkr_embed_dim=6,
        lang_embed_dim=6,
        num_conv_pos_embeddings=8,
        unit_hifi_gan_vocab_size=20,
        t2u_num_langs=0,
        t2u_max_new_tokens=25,
        t2u_offset_tgt_lang=0,
        vocoder_offset=0,
    ):
        self.parent = parent
        self.input_modality = input_modality

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

        self.vocab_size = vocab_size
        self.t2u_vocab_size = t2u_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.t2u_encoder_layers = t2u_encoder_layers
        self.t2u_decoder_layers = t2u_decoder_layers
        self.t2u_encoder_ffn_dim = t2u_encoder_ffn_dim
        self.t2u_decoder_ffn_dim = t2u_decoder_ffn_dim
        self.num_heads = num_heads
        self.num_attention_heads = num_heads

        self.vocoder_num_spkrs = vocoder_num_spkrs
        self.vocoder_num_langs = vocoder_num_langs
        self.upsample_initial_channel = upsample_initial_channel
        self.unit_embed_dim = unit_embed_dim
        self.spkr_embed_dim = spkr_embed_dim
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.lang_embed_dim = lang_embed_dim

        self.max_new_tokens = max_new_tokens

        self.unit_hifi_gan_vocab_size = unit_hifi_gan_vocab_size
        self.t2u_num_langs = t2u_num_langs
        self.t2u_max_new_tokens = t2u_max_new_tokens
        self.t2u_offset_tgt_lang = t2u_offset_tgt_lang
        self.vocoder_offset = vocoder_offset

    def prepare_config_and_inputs(self):
        if self.input_modality == "text":
            inputs = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1)
        else:
            inputs = ids_tensor([self.batch_size, self.seq_length, 160], self.vocab_size - 1).float()

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1)

        lm_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, inputs, decoder_input_ids, input_mask, lm_labels

    def get_config(self):
        return SeamlessM4TConfig(
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            vocab_size=self.vocab_size,
            t2u_vocab_size=self.t2u_vocab_size,
            hidden_size=self.hidden_size,
            speech_encoder_layers=self.num_heads,
            speech_encoder_intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            encoder_ffn_dim=self.encoder_ffn_dim,
            decoder_ffn_dim=self.decoder_ffn_dim,
            t2u_encoder_layers=self.t2u_encoder_layers,
            t2u_decoder_layers=self.t2u_decoder_layers,
            t2u_encoder_ffn_dim=self.t2u_encoder_ffn_dim,
            t2u_decoder_ffn_dim=self.t2u_decoder_ffn_dim,
            num_attention_heads=self.num_heads,
            encoder_attention_heads=self.num_heads,
            decoder_attention_heads=self.num_heads,
            t2u_encoder_attention_heads=self.num_heads,
            t2u_decoder_attention_heads=self.num_heads,
            speech_encoder_attention_heads=self.num_heads,
            unit_hifigan_vocab_vise=self.t2u_vocab_size,
            vocoder_num_spkrs=self.vocoder_num_spkrs,
            vocoder_num_langs=self.vocoder_num_langs,
            upsample_initial_channel=self.upsample_initial_channel,
            unit_embed_dim=self.unit_embed_dim,
            spkr_embed_dim=self.spkr_embed_dim,
            num_conv_pos_embeddings=self.num_conv_pos_embeddings,
            lang_embed_dim=self.lang_embed_dim,
            max_new_tokens=self.max_new_tokens,
            unit_hifi_gan_vocab_size=self.unit_hifi_gan_vocab_size,
            t2u_num_langs=self.t2u_num_langs,
            t2u_max_new_tokens=self.t2u_max_new_tokens,
            t2u_offset_tgt_lang=self.t2u_offset_tgt_lang,
            vocoder_offset=self.vocoder_offset,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            decoder_input_ids,
            input_mask,
            lm_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True

        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            decoder_input_ids,
            input_mask,
            lm_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_model(self, config, input_ids, decoder_input_ids, input_mask, labels):
        model = SeamlessM4TModel(config=config)
        model.to(torch_device)
        model.eval()
        if self.input_modality == "text":
            result = model(input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids)
            result = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        else:
            result = model(input_features=input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids)
            result = model(input_features=input_ids, decoder_input_ids=decoder_input_ids)
            self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

        decoder_output = result.logits
        decoder_past = result.past_key_values
        encoder_output = result.encoder_last_hidden_state

        if self.input_modality == "text":
            seq_length = self.seq_length
        else:
            # if speech, expected length has been subsampled.
            seq_length = model._compute_sub_sample_lengths_from_attention_mask(input_mask).max().item()

        self.parent.assertEqual(encoder_output.size(), (self.batch_size, seq_length, self.hidden_size))
        self.parent.assertEqual(decoder_output.size(), (self.batch_size, decoder_input_ids.shape[1], self.vocab_size))
        # There should be `num_layers` key value embeddings stored in decoder_past
        self.parent.assertEqual(len(decoder_past), config.decoder_layers)
        # There should be a self attn key, a self attn value, a cross attn key and a cross attn value stored in each decoder_past tuple
        self.parent.assertEqual(len(decoder_past[0]), 4)

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        decoder_input_ids,
        input_mask,
        lm_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        model = SeamlessM4TModel(config=config)
        model.to(torch_device)
        model.eval()

        # make sure no pad token in decoder_input_ids
        decoder_input_ids = torch.clamp(decoder_input_ids, config.pad_token_id + 1)

        # first forward pass
        outputs = model(
            input_ids, decoder_input_ids=decoder_input_ids, decoder_attention_mask=input_mask, use_cache=True
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            input_ids,
            decoder_input_ids=next_input_ids,
            decoder_attention_mask=next_attention_mask,
            output_hidden_states=True,
        )
        output_from_no_past = output_from_no_past["decoder_hidden_states"][0]
        output_from_past = model(
            input_ids,
            decoder_input_ids=next_tokens,
            decoder_attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["decoder_hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            decoder_input_ids,
            input_mask,
            lm_labels,
        ) = config_and_inputs

        input_name = "input_ids" if self.input_modality == "text" else "input_features"

        inputs_dict = {
            input_name: input_ids,
            "attention_mask": input_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": lm_labels,
        }
        return config, inputs_dict


@require_torch
class SeamlessM4TModelWithSpeechInputTest(ModelTesterMixin, unittest.TestCase):
    is_encoder_decoder = True
    fx_compatible = False
    test_missing_keys = False
    test_pruning = False
    test_model_parallel = False
    test_resize_embeddings = False
    test_headmasking = False
    test_torchscript = False

    all_model_classes = (
        (
            SeamlessM4TModel,
            SeamlessM4TForSpeechToSpeech,
            SeamlessM4TForSpeechToText,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (SeamlessM4TForSpeechToText,) if is_torch_available() else ()

    input_name = "input_features"

    def setUp(self):
        self.model_tester = SeamlessM4TModelTester(self, input_modality="speech")
        self.config_tester = ConfigTester(self, config_class=SeamlessM4TConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = SeamlessM4TModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def _get_input_ids_and_config(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict[self.input_name]

        # cut to half length & take max batch_size 3
        sequence_length = input_ids.shape[-1] // 2
        input_ids = input_ids[:batch_size, :sequence_length]

        # generate max 3 tokens
        max_length = input_ids.shape[-1] + 3
        if config.eos_token_id is not None and config.pad_token_id is None:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            if isinstance(config.eos_token_id, int):
                config.eos_token_id = [config.eos_token_id]
            config.pad_token_id = config.eos_token_id[0]

        attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.long)[:batch_size, :sequence_length]

        return config, input_ids.float(), attention_mask, max_length

    @staticmethod
    def _get_encoder_outputs(
        model, input_ids, attention_mask, output_attentions=None, output_hidden_states=None, num_interleave=1
    ):
        encoder = model.get_encoder()
        encoder_outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
            num_interleave, dim=0
        )
        input_ids = (
            torch.zeros(input_ids.shape[:2], dtype=torch.int64, layout=input_ids.layout, device=input_ids.device)
            + model._get_decoder_start_token_id()
        )
        attention_mask = None
        return encoder_outputs, input_ids, attention_mask

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "conv.weight",
                    "masked_spec_embed",
                    "codevectors",
                    "quantizer.weight_proj.weight",
                    "project_hid.weight",
                    "project_hid.bias",
                    "project_q.weight",
                    "project_q.bias",
                    "pos_bias_v",
                    "pos_bias_u",
                    "pointwise_conv1",
                    "pointwise_conv2",
                    "feature_projection.projection.weight",
                    "feature_projection.projection.bias",
                    "objective.weight",
                    "adapter",
                ]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    @unittest.skip(reason="SeamlessM4TSpeechEncoder doesn't have an embedding layer")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(
        reason="Expected missing keys serve when using SeamlessM4TForXXX.from_pretrained from a checkpoint saved by SeamlessM4TModel.save_pretrained."
    )
    def test_model_weights_reload_no_missing_tied_weights(self):
        pass

    @unittest.skip(
        reason="SeamlessM4TModel is base class but has actually a bigger architecture than seamlessM4T task-specific models."
    )
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="SeamlessM4TModel can takes input_ids or input_features")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="SeamlessM4T has no base model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def test_attention_outputs(self):
        # expected length is subsampled so need to change a bit this test
        if not self.has_attentions:
            self.skipTest(reason="Model does not output attentions")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        # no more chunk_length test

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # loss is at first position
                if "labels" in inputs_dict:
                    correct_outlen += 1  # loss is added to beginning
                if "past_key_values" in outputs:
                    correct_outlen += 1  # past_key_values have been returned

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

                sub_sampled_length = (
                    model._compute_sub_sample_lengths_from_attention_mask(inputs_dict["attention_mask"]).max().item()
                )
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        sub_sampled_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
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
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    @unittest.skip(
        reason="In training model, the first speech encoder layer is sometimes skipped. Training is not supported yet, so the test is ignored."
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass


@require_torch
class SeamlessM4TModelWithTextInputTest(
    ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    is_encoder_decoder = True
    fx_compatible = False
    test_missing_keys = False
    test_pruning = False
    test_model_parallel = False
    test_resize_embeddings = True
    test_headmasking = False
    test_torchscript = False

    all_model_classes = (
        (
            SeamlessM4TModel,
            SeamlessM4TForTextToSpeech,
            SeamlessM4TForTextToText,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (SeamlessM4TForTextToText,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "automatic-speech-recognition": SeamlessM4TForSpeechToText,
            "conversational": SeamlessM4TForTextToText,
            "feature-extraction": SeamlessM4TModel,
            "summarization": SeamlessM4TForTextToText,
            "text-to-audio": SeamlessM4TForTextToSpeech,
            "text2text-generation": SeamlessM4TForTextToText,
            "translation": SeamlessM4TForTextToText,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = SeamlessM4TModelTester(self, input_modality="text")
        self.config_tester = ConfigTester(self, config_class=SeamlessM4TConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = SeamlessM4TModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "conv.weight",
                    "masked_spec_embed",
                    "codevectors",
                    "quantizer.weight_proj.weight",
                    "project_hid.weight",
                    "project_hid.bias",
                    "project_q.weight",
                    "project_q.bias",
                    "pos_bias_v",
                    "pos_bias_u",
                    "pointwise_conv1",
                    "pointwise_conv2",
                    "feature_projection.projection.weight",
                    "feature_projection.projection.bias",
                    "objective.weight",
                    "adapter",
                ]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    @unittest.skip(
        reason="Expected missing keys serve when using SeamlessM4TForXXX.from_pretrained from a checkpoint saved by SeamlessM4TModel.save_pretrained."
    )
    def test_model_weights_reload_no_missing_tied_weights(self):
        pass

    @unittest.skip(reason="SeamlessM4TModel can take input_ids or input_features")
    def test_forward_signature(self):
        pass

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    @unittest.skip(
        reason="SeamlessM4TModel is base class but has actually a bigger architecture than seamlessM4T task-specific models."
    )
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="SeamlessM4T has no base model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        reason="In training model, the first encoder layer is sometimes skipped. Training is not supported yet, so the test is ignored."
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass


@require_torch
class SeamlessM4TGenerationTest(unittest.TestCase):
    # test that non-standard generation works
    # test generation of: SeamlessM4TModel, SeamlessM4TForSpeechToSpeech, SeamlessM4TForSpeechToText, SeamlessM4TForTextToSpeech

    def setUp(self):
        self.speech_model_tester = SeamlessM4TModelTester(self, input_modality="speech")
        self.text_model_tester = SeamlessM4TModelTester(self, input_modality="text")
        self.tmpdirname = tempfile.mkdtemp()

    def update_generation(self, model):
        lang_code_to_id = {
            "fra": 4,
            "eng": 4,
        }

        generation_config = copy.deepcopy(model.generation_config)

        generation_config.__setattr__("text_decoder_lang_to_code_id", lang_code_to_id)
        generation_config.__setattr__("t2u_lang_code_to_id", lang_code_to_id)
        generation_config.__setattr__("vocoder_lang_code_to_id", lang_code_to_id)

        generation_config._from_model_config = False

        model.generation_config = generation_config

    def prepare_text_input(self):
        config, inputs, decoder_input_ids, input_mask, lm_labels = self.text_model_tester.prepare_config_and_inputs()

        input_dict = {
            "input_ids": inputs,
            "attention_mask": input_mask,
            "tgt_lang": "eng",
            "num_beams": 2,
            "do_sample": True,
        }

        return config, input_dict

    def prepare_speech_input(self):
        config, inputs, decoder_input_ids, input_mask, lm_labels = self.speech_model_tester.prepare_config_and_inputs()

        input_dict = {
            "input_features": inputs,
            "attention_mask": input_mask,
            "tgt_lang": "fra",
            "num_beams": 2,
            "do_sample": True,
        }

        return config, input_dict

    def prepare_speech_and_text_input(self):
        config, inputs, decoder_input_ids, input_mask, lm_labels = self.speech_model_tester.prepare_config_and_inputs()

        input_speech = {
            "input_features": inputs,
            "attention_mask": input_mask,
            "tgt_lang": "fra",
            "num_beams": 2,
            "do_sample": True,
        }

        config, inputs, decoder_input_ids, input_mask, lm_labels = self.text_model_tester.prepare_config_and_inputs()

        input_text = {
            "input_ids": inputs,
            "attention_mask": input_mask,
            "tgt_lang": "eng",
            "num_beams": 2,
            "do_sample": True,
        }
        return config, input_speech, input_text

    def factory_generation_speech_test(self, model, inputs):
        set_seed(0)
        output = model.generate(**inputs)
        return output

    def test_speech_generation(self):
        config, input_speech, input_text = self.prepare_speech_and_text_input()

        model = SeamlessM4TModel(config=config)
        self.update_generation(model)
        model.save_pretrained(self.tmpdirname)
        model.to(torch_device)
        model.eval()

        output_original_text = self.factory_generation_speech_test(model, input_text)
        output_original_speech = self.factory_generation_speech_test(model, input_speech)

        state_dict = model.state_dict()

        text_model = SeamlessM4TForTextToSpeech.from_pretrained(self.tmpdirname)
        self.update_generation(text_model)
        text_model.to(torch_device)
        text_model.eval()

        output_text = self.factory_generation_speech_test(model, input_text)

        speech_model = SeamlessM4TForSpeechToSpeech.from_pretrained(self.tmpdirname)
        self.update_generation(speech_model)
        speech_model.to(torch_device)
        speech_model.eval()

        for name, tensor in speech_model.state_dict().items():
            right_tensor = state_dict.get(name)
            self.assertEqual(tensor.tolist(), right_tensor.tolist(), f"Tensor {name}")

        output_speech = self.factory_generation_speech_test(model, input_speech)

        # test same text output from input text
        self.assertListEqual(output_original_text[0].ravel().tolist(), output_text[0].ravel().tolist())
        self.assertListEqual(output_original_text[1].ravel().tolist(), output_text[1].ravel().tolist())

        # test same speech output from input text
        # assertTrue because super long list makes this hang in case of failure
        self.assertTrue(
            output_original_speech[0].ravel().tolist() == output_speech[0].ravel().tolist(),
            "Speech generated was different",
        )
        self.assertTrue(
            output_original_speech[1].ravel().tolist() == output_speech[1].ravel().tolist(),
            "Speech generated was different",
        )

    def test_text_generation(self):
        config, input_speech, input_text = self.prepare_speech_and_text_input()

        # to return speech
        input_speech["generate_speech"] = False
        input_text["generate_speech"] = False

        model = SeamlessM4TModel(config=config)
        self.update_generation(model)
        model.save_pretrained(self.tmpdirname)
        model.to(torch_device)
        model.eval()

        output_original_text = self.factory_generation_speech_test(model, input_text)
        output_original_speech = self.factory_generation_speech_test(model, input_speech)

        # other models don't need it
        input_speech.pop("generate_speech")
        input_text.pop("generate_speech")

        state_dict = model.state_dict()

        text_model = SeamlessM4TForTextToText.from_pretrained(self.tmpdirname)
        self.update_generation(text_model)
        text_model.to(torch_device)
        text_model.eval()

        for name, tensor in text_model.state_dict().items():
            right_tensor = state_dict.get(name)
            self.assertEqual(tensor.tolist(), right_tensor.tolist())

        output_text = self.factory_generation_speech_test(text_model, input_text)

        speech_model = SeamlessM4TForSpeechToText.from_pretrained(self.tmpdirname)

        for name, tensor in speech_model.state_dict().items():
            right_tensor = state_dict.get(name)
            self.assertEqual(tensor.tolist(), right_tensor.tolist(), f"Tensor {name}")

        self.update_generation(speech_model)
        speech_model.to(torch_device)
        speech_model.eval()

        output_speech = self.factory_generation_speech_test(speech_model, input_speech)

        # test same text output from input text
        self.assertListEqual(output_original_text[0].ravel().tolist(), output_text.ravel().tolist())

        # test same speech output from input text
        self.assertListEqual(output_original_speech[0].ravel().tolist(), output_speech.ravel().tolist())

    def test_generation(self):
        config, input_speech, input_text = self.prepare_speech_and_text_input()

        input_speech["num_beams"] = 3
        input_speech["do_sample"] = True
        input_speech["num_return_sequences"] = 3

        input_text["num_beams"] = 3
        input_text["do_sample"] = True
        input_text["num_return_sequences"] = 3

        for model_class in [SeamlessM4TForSpeechToSpeech, SeamlessM4TForSpeechToText, SeamlessM4TModel]:
            model = model_class(config=config)
            self.update_generation(model)
            model.to(torch_device)
            model.eval()

            output = model.generate(**input_speech)
            output = output[0] if isinstance(output, tuple) else output

            self.assertEqual(output.shape[0], 3 * input_speech["input_features"].shape[0])

        for model_class in [SeamlessM4TForTextToSpeech, SeamlessM4TForTextToText, SeamlessM4TModel]:
            model = model_class(config=config)
            self.update_generation(model)
            model.to(torch_device)
            model.eval()

            output = model.generate(**input_text)

            output = output[0] if isinstance(output, tuple) else output

            self.assertEqual(output.shape[0], 3 * input_text["input_ids"].shape[0])


@require_torch
class SeamlessM4TModelIntegrationTest(unittest.TestCase):
    repo_id = "facebook/hf-seamless-m4t-medium"

    def assertListAlmostEqual(self, list1, list2, tol=1e-3):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, delta=tol)

    @cached_property
    def processor(self):
        return SeamlessM4TProcessor.from_pretrained(self.repo_id)

    @cached_property
    def input_text(self):
        # corresponds to "C'est un test." with seamlessM4T_medium checkpoint

        input_ids = torch.tensor([[256057, 152, 248116, 354, 159, 7356, 248075, 3]])  # fmt: skip

        input_ids = input_ids.to(torch_device)

        attention_mask = torch.ones_like(input_ids).to(torch_device)

        inputs = {
            "attention_mask": attention_mask,
            "input_ids": input_ids,
        }

        return inputs

    @cached_property
    def input_audio(self):
        set_seed(0)
        seq_len = 20000
        sampling_rate = 16000
        input_features = torch.rand((2, seq_len))

        return self.processor(audios=[input_features.tolist()], sampling_rate=sampling_rate, return_tensors="pt").to(
            torch_device
        )

    def factory_test_task(self, class1, class2, inputs, class1_kwargs, class2_kwargs):
        model1 = class1.from_pretrained(self.repo_id).to(torch_device)
        model2 = class2.from_pretrained(self.repo_id).to(torch_device)

        set_seed(0)
        output_1 = model1.generate(**inputs, **class1_kwargs)
        set_seed(0)
        output_2 = model2.generate(**inputs, **class2_kwargs)

        for key in output_1:
            if isinstance(output_1[key], torch.Tensor):
                if len(output_1[key].shape) == 0:
                    self.assertEqual(output_1[key].item(), output_2[key].item())
                else:
                    self.assertListAlmostEqual(output_1[key].squeeze().tolist(), output_2[key].squeeze().tolist())

    @slow
    def test_to_eng_text(self):
        model = SeamlessM4TModel.from_pretrained(self.repo_id).to(torch_device)

        # test text - tgt lang: eng

        expected_text_tokens = [3, 256047, 3291, 248116, 248066, 9, 7356, 248075, 3]  # fmt: skip

        # fmt: off
        expected_unit_tokens = [
            2,10051,8980,8212,949,1270,4311,1123,5918,2333,5311,3882,2415,5284,1123,612,8816,6370,5386,7334,4345,5645,
            9437,5748,1378,9818,4319,7968,7375,2909,9119,5151,8728,5335,3896,4013,8939,8885,6048,9530,3167,5833,1072,693,
            431,9867,364,7909,4608,5938,1889,9984,7947,4944,6171,3767,9861,9169,1187,8365,4571,7635,7784,7635,800,2393,
            32,5380,5852,8289,2530,2762,1833,2056,3553,4641,3553,5683,370,2288,1344,1518,7534,703,8359,7699,2
        ]
        # fmt: on

        expected_wav_slice = [-3e-05, -0.0004, -0.00037, -0.00013, -6e-05, 0.00012, -0.00016, 0.00025, 7e-05, -3e-05]  # fmt: skip

        set_seed(0)
        output = model.generate(**self.input_text, num_beams=1, tgt_lang="eng", return_intermediate_token_ids=True)

        self.assertListEqual(expected_text_tokens, output.sequences.squeeze().tolist())
        # FOR NOW, only first units correspondance
        self.assertListEqual(expected_unit_tokens[:10], output.unit_sequences.squeeze().tolist()[:10])

        self.assertListAlmostEqual(expected_wav_slice, output.waveform.squeeze().tolist()[50:60])

    @slow
    def test_to_swh_text(self):
        model = SeamlessM4TModel.from_pretrained(self.repo_id).to(torch_device)

        # test text - tgt lang: swh

        expected_text_tokens = [3, 256168, 1665, 188589, 7040, 248075, 3]  # fmt: skip

        # fmt: off
        expected_unit_tokens = [
            2,10071,5729,9995,3089,7546,1204,1721,2532,4340,5623,3496,432,7730,9096,7677,3143,8211,6447,8399,4248,3565,
            4529,7700,9308,217,6476,3485,9667,3194,8476,4923,5593,1148,4466,7416,4872,463,4872,253,2348,4640,3450,2133,
            6318,2806,817,7613,2698,6563,8712,8344,9286,6878,6387,4281,6387,640,6387,3200,640,8355,640,6708,979,1738,2
        ]
        # fmt: on

        expected_wav_slice = [1e-05, -7e-05, -4e-05, -4e-05, -6e-05, -9e-05, -0.0001, -2e-05, -7e-05, -2e-05]  # fmt: skip

        set_seed(0)
        output = model.generate(**self.input_text, num_beams=1, tgt_lang="swh", return_intermediate_token_ids=True)

        self.assertListEqual(expected_text_tokens, output.sequences.squeeze().tolist())
        self.assertListEqual(expected_unit_tokens[:10], output.unit_sequences.squeeze().tolist()[:10])

        self.assertListAlmostEqual(expected_wav_slice, output.waveform.squeeze().tolist()[50:60])

    @slow
    def test_to_rus_speech(self):
        model = SeamlessM4TModel.from_pretrained(self.repo_id).to(torch_device)

        # test audio - tgt lang: rus

        expected_text_tokens = [3, 256147, 1197, 73565, 3413, 537, 233331, 248075, 3]  # fmt: skip

        # fmt: off
        expected_unit_tokens = [
            2, 10067, 5729, 4798, 9631, 8378, 4446, 2393, 6901, 5983, 2817, 4629, 8532, 1991, 2931, 8576, 8857, 5936, 4317,
            9000, 7740, 7995, 1225, 5980, 6094, 1420, 5373, 8771, 6600, 4487, 7029, 3630, 6740, 4870, 1483, 3003, 5585, 5511,
            7465, 3222, 32, 6272, 1950, 3120, 5368, 639, 3713, 5935, 7943, 567, 6129, 6822, 1226, 5063, 9878, 7756, 8825, 1078, 5943,
            457, 9282, 9668, 817, 7613, 2698, 6563, 8712, 8704, 9286, 8704, 6387, 4281, 6387, 640, 3200, 6387, 640, 8355, 6708, 979, 1738, 2
        ]
        # fmt: on

        expected_wav_slice = [0.00013, 0.00012, 0.00014, 3e-05, 0.0, -6e-05, -0.00018, -0.00016, -0.00021, -0.00018]  # fmt: skip

        set_seed(0)
        output = model.generate(**self.input_audio, num_beams=1, tgt_lang="rus", return_intermediate_token_ids=True)

        self.assertListEqual(expected_text_tokens, output.sequences.squeeze().tolist())
        self.assertListEqual(expected_unit_tokens[:10], output.unit_sequences.squeeze().tolist()[:10])

        self.assertListAlmostEqual(expected_wav_slice, output.waveform.squeeze().tolist()[50:60])

    @slow
    def test_text_to_text_model(self):
        kwargs1 = {"tgt_lang": "eng", "return_intermediate_token_ids": True, "generate_speech": False}
        kwargs2 = {
            "tgt_lang": "eng",
            "output_hidden_states": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        self.factory_test_task(SeamlessM4TModel, SeamlessM4TForTextToText, self.input_text, kwargs1, kwargs2)

    @slow
    def test_speech_to_text_model(self):
        kwargs1 = {"tgt_lang": "eng", "return_intermediate_token_ids": True, "generate_speech": False}
        kwargs2 = {
            "tgt_lang": "eng",
            "output_hidden_states": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        self.factory_test_task(SeamlessM4TModel, SeamlessM4TForSpeechToText, self.input_audio, kwargs1, kwargs2)

    @slow
    def test_speech_to_speech_model(self):
        kwargs1 = {"tgt_lang": "eng", "return_intermediate_token_ids": True}
        self.factory_test_task(SeamlessM4TModel, SeamlessM4TForSpeechToSpeech, self.input_audio, kwargs1, kwargs1)

    @slow
    def test_text_to_speech_model(self):
        kwargs1 = {"tgt_lang": "eng", "return_intermediate_token_ids": True}

        self.factory_test_task(SeamlessM4TModel, SeamlessM4TForTextToSpeech, self.input_text, kwargs1, kwargs1)
