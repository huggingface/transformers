# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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


import unittest

from transformers import SeamlessM4TConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


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


class SeamlessM4TModelTester:
    def __init__(
        self,
        parent,
        input_modality="speech",
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
        vocab_size=24,
        unit_vocab_size=24,
        hidden_size=24,
        num_hidden_layers=2,
        intermediate_size=24,
        max_position_embeddings=2048,
        encoder_layers=2,
        decoder_layers=2,
        encoder_ffn_dim=24,
        decoder_ffn_dim=24,
        t2u_encoder_layers=2,
        t2u_decoder_layers=2,
        t2u_encoder_ffn_dim=24,
        t2u_decoder_ffn_dim=24,
        num_heads=6,
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
        self.unit_vocab_size = unit_vocab_size
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

    def prepare_config_and_inputs(self):
        if self.input_modality == "text":
            inputs = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        else:
            inputs = ids_tensor([self.batch_size, self.seq_length, 160], self.vocab_size).float()

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        # TODO: keep?
        # if self.use_labels:
        #    sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        #    token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
        #    choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, inputs, input_mask, lm_labels

    def get_config(self):
        return SeamlessM4TConfig(
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            vocab_size=self.vocab_size,
            unit_vocab_size=self.unit_vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
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
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            lm_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True

        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            input_mask,
            lm_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = SeamlessM4TModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(
            input_ids,
        )
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # def create_and_check_for_causal_lm(
    #    self,
    #    config,
    #    input_ids,
    #    input_mask,
    # ):
    #    model = SeamlessM4TForCausalLM(config=config)
    #    model.to(torch_device)
    #    model.eval()
    #    result = model(input_ids, attention_mask=input_mask, , labels=token_labels)
    #    self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = SeamlessM4TModel(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            input_ids=next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            input_ids=next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

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
            input_mask,
            lm_labels,
        ) = config_and_inputs

        input_name = "input_ids" if self.input_modality == "text" else "input_features"

        inputs_dict = {input_name: input_ids, "attention_mask": input_mask, "labels": lm_labels}
        return config, inputs_dict


@require_torch
class SeamlessM4TModelWithSpeechInputTest(ModelTesterMixin, unittest.TestCase):
    is_encoder_decoder = True
    fx_compatible = False
    test_missing_keys = False
    test_pruning = False
    test_model_parallel = True
    test_resize_embeddings = True

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

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

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


@require_torch
class SeamlessM4TModelWithTextInputTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    is_encoder_decoder = True
    fx_compatible = False
    test_missing_keys = False
    test_pruning = False
    test_model_parallel = True
    test_resize_embeddings = True

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


@require_torch
class SeamlessM4TModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = SeamlessM4TModel.from_pretrained("meta-private/m4t_large")
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        # TODO Replace vocab size
        vocab_size = 32000

        expected_shape = torch.Size((1, 6, vocab_size))
        self.assertEqual(output.shape, expected_shape)

        # TODO Replace values below with what was printed above.
        expected_slice = torch.tensor(
            [[[-0.0483, 0.1188, -0.0313], [-0.0606, 0.1435, 0.0199], [-0.0235, 0.1519, 0.0175]]]
        )

        # sentence: "This is something to be translated in French"
        # fmt: off
        # fmt:on

        # beam_size = 1
        # fmt: off
        # fmt: on

        # fmt: off
        # fmt: on

        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
