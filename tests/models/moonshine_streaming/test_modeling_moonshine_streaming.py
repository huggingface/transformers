# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch MoonshineStreaming model."""

import copy
import unittest

from transformers import MoonshineStreamingConfig, MoonshineStreamingEncoderConfig, is_torch_available
from transformers.testing_utils import Expectations, cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        MoonshineStreamingForConditionalGeneration,
        MoonshineStreamingModel,
    )

from datasets import load_dataset


class MoonshineStreamingModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=1040,
        is_training=False,
        use_labels=False,
        vocab_size=147,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        decoder_start_token_id=85,
        bos_token_id=98,
        eos_token_id=98,
        pad_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.decoder_start_token_id = decoder_start_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        decoder_input_ids = torch.tensor(self.batch_size * [[self.decoder_start_token_id]], device=torch_device)
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id)

        config = self.get_config()

        return config, input_values, attention_mask, decoder_input_ids, decoder_attention_mask

    def get_config(self):
        encoder_config = MoonshineStreamingEncoderConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,
            head_dim=self.head_dim,
        )
        return MoonshineStreamingConfig(
            encoder_config=encoder_config,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            head_dim=self.head_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            decoder_start_token_id=self.decoder_start_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def check_output_attentions(self, config, input_values, attention_mask):
        model = MoonshineStreamingModel(config=config)
        model.to(torch_device)
        model.train()

        outputs = model(input_values, attention_mask=attention_mask, output_attentions=True)
        self.parent.assertTrue(len(outputs.attentions) > 0)

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask, decoder_input_ids, decoder_attention_mask = (
            self.prepare_config_and_inputs()
        )
        inputs_dict = {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict


@require_torch
class MoonshineStreamingModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (MoonshineStreamingModel, MoonshineStreamingForConditionalGeneration) if is_torch_available() else ()
    )
    # Doesn't run generation tests. TODO (eustache): remove this line and then make CI green
    all_generative_model_classes = ()
    pipeline_model_mapping = (
        {
            "automatic-speech-recognition": MoonshineStreamingForConditionalGeneration,
            "feature-extraction": MoonshineStreamingModel,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = MoonshineStreamingModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MoonshineStreamingConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_can_init_all_missing_weights(self):
        self.skipTest("MoonshineStreaming uses special parameter initialization that conflicts with this test")

    def test_init_weights_can_init_buffers(self):
        self.skipTest("MoonshineStreaming uses special buffer initialization that conflicts with this test")

    # MoonshineStreaming test model's pos_emb layer is large relative to total model size,
    # requiring higher GPU allocation percentage to enable device splitting
    model_split_percents = [0.5, 0.9, 0.95]

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", 1)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", 1)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()

            subsampled_encoder_seq_length = model._get_feat_extract_output_lengths(encoder_seq_length)
            subsampled_encoder_key_length = model._get_feat_extract_output_lengths(encoder_key_length)

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            config.encoder_config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )
            out_len = len(outputs)

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
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    decoder_seq_length,
                    subsampled_encoder_key_length,
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

            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
            else:
                seq_length = self.model_tester.seq_length

            subsampled_seq_length = model._get_feat_extract_output_lengths(seq_length)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [subsampled_seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)

                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", 1)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            config.encoder_config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # Copied from tests.models.whisper.test_modeling_whisper.WhisperModelTest.test_inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            decoder_input_ids = inputs.pop("decoder_input_ids", None)
            inputs.pop("decoder_attention_mask", None)

            wte = model.get_input_embeddings()
            inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    # Copied from tests.models.whisper.test_modeling_whisper.WhisperModelTest.test_resize_tokens_embeddings
    def test_resize_tokens_embeddings(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is False")

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # make sure that decoder_input_ids are resized
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    # Copied from tests.models.whisper.test_modeling_whisper.WhisperModelTest.test_resize_embeddings_untied
    def test_resize_embeddings_untied(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is False")

        original_config.tie_word_embeddings = False

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            self.skipTest(reason="Model cannot untie embeddings")

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)
            model.eval()

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))


@require_torch
class MoonshineStreamingModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.processor_tiny = AutoProcessor.from_pretrained("UsefulSensors/moonshine-streaming-tiny")
        self.processor_small = AutoProcessor.from_pretrained("UsefulSensors/moonshine-streaming-small")
        self.processor_medium = AutoProcessor.from_pretrained("UsefulSensors/moonshine-streaming-medium")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id")[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @slow
    def test_tiny_logits_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-tiny")
        model.to(torch_device)

        inputs = self.processor_tiny(self._load_datasamples(1), sampling_rate=16000)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            -13.847891807556152, -0.18819725513458252, 3.1453802585601807, -13.759804725646973, -13.689135551452637,
            -13.750009536743164, -13.690473556518555, -13.681711196899414, -13.769899368286133, -13.692444801330566,
            -13.809157371520996, -13.810665130615234, -13.652420043945312, -13.789128303527832, -13.746649742126465,
            -13.74869155883789, -13.79692268371582, -13.63906192779541, -13.665060997009277, -13.634946823120117,
            -13.711505889892578, -13.777567863464355, -13.721321105957031, -13.677959442138672, -13.754849433898926,
            -13.712194442749023, -13.79233169555664, -13.687705039978027, -13.664924621582031, -13.779203414916992,
        ])
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][0, :30].cpu(), EXPECTED_LOGITS, rtol=2e-4, atol=2e-4)

    @slow
    def test_small_logits_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-small")
        model.to(torch_device)

        inputs = self.processor_small(self._load_datasamples(1), sampling_rate=16000)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            -9.193448066711426, -1.3106095790863037, 2.4847524166107178, -9.474504470825195, -9.443048477172852,
            -9.465521812438965, -9.475011825561523, -9.474539756774902, -9.452878952026367, -9.46949577331543,
            -9.46340560913086, -9.48450756072998, -9.512656211853027, -9.460539817810059, -9.464164733886719,
            -9.46074104309082, -9.420138359069824, -9.48065185546875, -9.467584609985352, -9.43082332611084,
            -9.467816352844238, -9.473931312561035, -9.462691307067871, -9.438430786132812, -9.448503494262695,
            -9.438905715942383, -9.440755844116211, -9.487390518188477, -9.487754821777344, -9.472284317016602,
        ])
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][0, :30].cpu(), EXPECTED_LOGITS, rtol=1e-4, atol=1e-4)

    @slow
    def test_medium_logits_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-medium")
        model.to(torch_device)

        inputs = self.processor_medium(self._load_datasamples(1), sampling_rate=16000)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            -9.380514144897461, -1.8016688823699951, 1.309783935546875, -9.992443084716797, -10.047298431396484,
            -9.993546485900879, -10.00343132019043, -10.052844047546387, -10.095193862915039, -9.937813758850098,
            -9.995306968688965, -10.06312370300293, -10.039563179016113, -10.00948715209961, -10.04725170135498,
            -10.08010196685791, -10.043283462524414, -10.06112289428711, -9.989591598510742, -10.034473419189453,
            -9.958343505859375, -9.956878662109375, -10.006301879882812, -10.032047271728516, -9.969188690185547,
            -10.00571060180664, -10.043065071105957, -9.983331680297852, -9.988570213317871, -9.935394287109375,
        ])
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][0, :30].cpu(), EXPECTED_LOGITS, rtol=1e-4, atol=1e-4)

    @slow
    def test_tiny_logits_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-tiny")
        model.to(torch_device)

        inputs = self.processor_tiny(self._load_datasamples(4), sampling_rate=16000)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)
        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                [-12.441858291625977, -0.2812096178531647, 2.7568106651306152, -12.284578323364258, -12.205985069274902, -12.262890815734863, -12.224806785583496, -12.220057487487793, -12.314021110534668, -12.228297233581543],
                [-13.319320678710938, -3.6359996795654297, 4.0685296058654785, -13.046940803527832, -13.122637748718262, -13.096488952636719, -13.141905784606934, -13.038910865783691, -13.136741638183594, -13.037278175354004],
                [-10.126669883728027, -4.161841869354248, 4.4407429695129395, -10.040196418762207, -10.065054893493652, -10.001801490783691, -9.991734504699707, -10.037150382995605, -10.0549898147583, -10.101166725158691],
                [-11.697093963623047, -3.0441789627075195, 3.8363659381866455, -11.45719051361084, -11.495401382446289, -11.519722938537598, -11.482342720031738, -11.529292106628418, -11.5482177734375, -11.483217239379883],
            ],
        )
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][:, :10].cpu(), EXPECTED_LOGITS, rtol=2e-4, atol=2e-4)

    @slow
    def test_small_logits_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-small")
        model.to(torch_device)

        inputs = self.processor_small(self._load_datasamples(4), sampling_rate=16000)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = Expectations(
            {
                ("xpu", None): torch.tensor(
                    [
                        [-9.5963, -1.2973, 2.8171, -9.8262, -9.8024, -9.8025, -9.8129, -9.8202, -9.8017, -9.8099],
                        [-9.6030, 0.3276, 3.0865, -9.7542, -9.8030, -9.8325, -9.7853, -9.7509, -9.8279, -9.8164],
                        [-10.2473, -0.4238, 3.1180, -9.9895, -10.0012, -10.0405, -9.9965, -10.0520, -9.9861, -10.0361],
                        [-9.9825, -1.4059, 3.5391, -9.4338, -9.4446, -9.4975, -9.4524, -9.4573, -9.4328, -9.4394],
                    ]
                ),
                (None, None): torch.tensor(
                    [
                        [-9.5963, -1.2973, 2.8171, -9.8262, -9.8024, -9.8025, -9.8129, -9.8202, -9.8017, -9.8099],
                        [-9.6030, 0.3276, 3.0865, -9.7542, -9.8030, -9.8325, -9.7853, -9.7509, -9.8279, -9.8164],
                        [-10.2473, -0.4232, 3.1180, -9.9895, -10.0012, -10.0405, -9.9965, -10.0520, -9.9861, -10.0361],
                        [-9.9825, -1.4063, 3.5391, -9.4338, -9.4446, -9.4975, -9.4524, -9.4573, -9.4328, -9.4394],
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][:, :10].cpu(), EXPECTED_LOGITS, rtol=2e-4, atol=2e-4)

    @slow
    def test_medium_logits_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-medium")
        model.to(torch_device)

        inputs = self.processor_medium(self._load_datasamples(4), sampling_rate=16000)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = Expectations(
            {
                ("xpu", None): torch.tensor(
                    [
                        [-9.4235, -1.6021, 1.3190, -10.0322, -10.0858, -10.0422, -10.0573, -10.0898, -10.1419, -10.0034],
                        [-9.8914, -2.2688, 2.4474, -10.1934, -10.2570, -10.1845, -10.2231, -10.2922, -10.3259, -10.2566],
                        [-9.3967, -0.7291, 2.2988, -9.8157, -9.8541, -9.8216, -9.8118, -9.8388, -9.8544, -9.8559],
                        [-8.9188, -0.6991, 1.3242, -8.9318, -9.0168, -8.9296, -8.9460, -8.9843, -8.9837, -8.9457],
                    ]
                ),
                (None, None): torch.tensor(
                    [
                        [-9.4235, -1.6021, 1.3190, -10.0322, -10.0858, -10.0422, -10.0573, -10.0898, -10.1419, -10.0034],
                        [-9.8914, -2.2688, 2.4474, -10.1934, -10.2570, -10.1845, -10.2231, -10.2922, -10.3260, -10.2566],
                        [-9.3967, -0.7291, 2.2995, -9.8157, -9.8541, -9.8216, -9.8118, -9.8388, -9.8544, -9.8559],
                        [-8.9188, -0.6991, 1.3242, -8.9318, -9.0168, -8.9296, -8.9460, -8.9843, -8.9837, -8.9457],
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][:, :10].cpu(), EXPECTED_LOGITS, rtol=2e-4, atol=2e-4)

    @slow
    def test_tiny_generation_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-tiny")
        model.to(torch_device)

        audio_array = self._load_datasamples(1)
        inputs = self.processor_tiny(audio_array, sampling_rate=16000)
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_tiny.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = "Mr. Quilter is the apostle of the Middle Classes, and we are glad to"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_small_generation_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-small")
        model.to(torch_device)

        audio_array = self._load_datasamples(1)
        inputs = self.processor_small(audio_array, sampling_rate=16000)
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_small.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = "Mister Quilter is the apostle of the middle classes, and we are glad to welcome"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_medium_generation_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-medium")
        model.to(torch_device)

        audio_array = self._load_datasamples(1)
        inputs = self.processor_medium(audio_array, sampling_rate=16000)
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_medium.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = "Mister Quilter is the apostle of the middle classes, and we are glad to welcome"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_tiny_generation_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-tiny")
        model.to(torch_device)

        audio_array = self._load_datasamples(4)
        inputs = self.processor_tiny(audio_array, sampling_rate=16000)
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_tiny.batch_decode(generated_ids, skip_special_tokens=True)

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            "Mr. Quilter is the apostle of the Middle Classes, and we are glad to",
            "Nor is Mr. Quilter's manner less interesting than his matter.",
            "He tells us that at this festive season of the year, with Christmas and a roast be",
            "He has grieved doubts whether Sir Frederick Layton's work is really Greek after all",
        ]
        # fmt: on

        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_small_generation_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-small")
        model.to(torch_device)

        audio_array = self._load_datasamples(4)
        inputs = self.processor_small(audio_array, sampling_rate=16000)
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_small.batch_decode(generated_ids, skip_special_tokens=True)

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            "Mister Quilter is the apostle of the middle classes, and we are glad to welcome",
            "Nor is Mister Quilter's manner less interesting than his matter.",
            "He tells us that at this festive season of the year, with Christmas and roast beef",
            "He has grave doubts whether Sir Frederick Layton's work is really Greek after all,",
        ]
        # fmt: on

        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_medium_generation_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-medium")
        model.to(torch_device)

        audio_array = self._load_datasamples(4)
        inputs = self.processor_medium(audio_array, sampling_rate=16000)
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_medium.batch_decode(generated_ids, skip_special_tokens=True)

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            "Mister Quilter is the apostle of the middle classes, and we are glad to welcome",
            "Nor is Mister Quilter's manner less interesting than his matter.",
            "He tells us that at this festive season of the year, with Christmas and roast beef",
            "He has grave doubts whether Sir Frederick Leighton's work is really Greek after all,",
        ]
        # fmt: on

        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)
