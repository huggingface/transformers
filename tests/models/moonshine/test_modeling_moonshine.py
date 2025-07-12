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
"""Testing suite for the PyTorch Moonshine model."""

import copy
import unittest

from transformers import MoonshineConfig, is_torch_available
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
        MoonshineForConditionalGeneration,
        MoonshineModel,
    )

from datasets import load_dataset


class MoonshineModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=1000,
        is_training=False,
        use_labels=False,
        vocab_size=147,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        encoder_hidden_act="gelu",
        decoder_hidden_act="silu",
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
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.encoder_hidden_act = encoder_hidden_act
        self.decoder_hidden_act = decoder_hidden_act
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
        return MoonshineConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            encoder_num_hidden_layers=self.num_hidden_layers,
            decoder_num_hidden_layers=self.num_hidden_layers,
            encoder_num_attention_heads=self.num_attention_heads,
            decoder_num_attention_heads=self.num_attention_heads,
            encoder_num_key_value_heads=self.num_key_value_heads,
            decoder_num_key_value_heads=self.num_key_value_heads,
            encoder_hidden_act=self.encoder_hidden_act,
            decoder_hidden_act=self.decoder_hidden_act,
            decoder_start_token_id=self.decoder_start_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    def check_output_attentions(self, config, input_values, attention_mask):
        model = MoonshineModel(config=config)
        model.config.layerdrop = 1.0
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
class MoonshineModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MoonshineModel, MoonshineForConditionalGeneration) if is_torch_available() else ()
    # Doesn't run generation tests. TODO (eustache): remove this line and then make CI green
    all_generative_model_classes = ()
    pipeline_model_mapping = (
        {
            "automatic-speech-recognition": MoonshineForConditionalGeneration,
            "feature-extraction": MoonshineModel,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_headmasking = False

    def setUp(self):
        self.model_tester = MoonshineModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MoonshineConfig)

    @unittest.skip("failing. Will fix only when the community opens an issue for it.")
    def test_torchscript_output_hidden_state(self):
        pass

    @unittest.skip("failing. Will fix only when the community opens an issue for it.")
    def test_torchscript_simple(self):
        pass

    def test_config(self):
        self.config_tester.run_common_tests()

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

    # Copied from tests.models.whisper.test_modeling_whisper.WhisperModelTest.test_hidden_states_output
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
class MoonshineModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.processor_tiny = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
        self.processor_base = AutoProcessor.from_pretrained("UsefulSensors/moonshine-base")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id")[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @slow
    def test_tiny_logits_single(self):
        model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny")
        model.to(torch_device)

        inputs = self.processor_tiny(self._load_datasamples(1), return_tensors="pt")
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        expectations = Expectations(
            {
                (None, None): [-9.1106, 4.5542, 6.3892, -6.8139, -7.2456, -7.9074, -7.2839, -7.6043, -8.0384, -7.8351, -7.3867, -7.2450, -7.7420, -7.3912, -7.3866, -7.6979, -7.6420, -7.0504, -7.3979, -7.2483, -8.0796, -7.3300, -7.3672, -6.8765, -7.6876, -7.2682, -6.9866, -6.7457, -7.6855, -7.3050],
                ("cuda", 8): [-9.1107, 4.5538, 6.3902, -6.8141, -7.2459, -7.9076, -7.2842, -7.6045, -8.0387, -7.8354, -7.3869, -7.2453, -7.7423, -7.3914, -7.3869, -7.6982, -7.6422, -7.0507, -7.3982, -7.2486, -8.0798, -7.3302, -7.3675, -6.8769, -7.6878, -7.2684, -6.9868, -6.7459, -7.6858, -7.3052],
            }
        )
        EXPECTED_LOGITS = torch.tensor(expectations.get_expectation()).to(torch_device)
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][0, :30], EXPECTED_LOGITS, rtol=2e-4, atol=2e-4)

    @slow
    def test_base_logits_single(self):
        model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-base")
        model.to(torch_device)

        inputs = self.processor_base(self._load_datasamples(1), return_tensors="pt")
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            -6.7336, 1.9482, 5.2448, -8.0277, -7.9167, -7.8956, -7.9649, -7.9348, -8.1312, -8.0616,
            -8.1070, -7.7696, -7.8809, -7.9450, -8.1013, -7.8177, -7.8598, -7.8257, -7.8729, -7.9657,
            -7.9310, -8.1024, -7.8699, -7.8231, -8.0752, -7.9764, -7.8127, -8.0536, -7.9492, -7.9290,
        ])
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][0, :30].cpu(), EXPECTED_LOGITS, rtol=1e-4, atol=1e-4)

    @slow
    def test_tiny_logits_batch(self):
        model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny")
        model.to(torch_device)

        inputs = self.processor_tiny(self._load_datasamples(4), return_tensors="pt", padding=True)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)
        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                [-8.5966, 4.8608, 5.8849, -6.6183, -7.0378, -7.7121, -7.0640, -7.3839, -7.8330, -7.6116],
                [-4.3147, -2.4953, 8.4924, -6.4803, -7.0949, -6.7498, -6.1081, -6.6481, -6.9866, -6.5916],
                [-10.0088, 3.2862, 0.7342, -6.5559, -6.8514, -6.5309, -6.4173, -6.9485, -6.6215, -6.6230],
                [-11.1002, 3.9398, 0.6674, -5.0146, -5.3936, -5.4099, -5.2236, -5.4404, -5.2200, -5.2702],
            ],
        )
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][:, :10].cpu(), EXPECTED_LOGITS, rtol=2e-4, atol=2e-4)

    @slow
    def test_base_logits_batch(self):
        model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-base")
        model.to(torch_device)

        inputs = self.processor_base(self._load_datasamples(4), return_tensors="pt", padding=True)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                [-6.3602, 1.8383, 5.2615, -7.9576, -7.8442, -7.8238, -7.9014, -7.8645, -8.0550, -7.9963],
                [-6.1725, -0.6274, 8.1798, -6.8570, -6.8078, -6.7915, -6.9099, -6.8980, -6.9760, -6.8264],
                [-7.3186, 3.1192, 3.8938, -5.7208, -5.8429, -5.7610, -5.9997, -5.8213, -5.8616, -5.8720],
                [-7.3432, 1.0402, 3.9912, -5.4177, -5.4890, -5.4573, -5.6516, -5.4776, -5.5079, -5.5391],
            ]
        )
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][:, :10].cpu(), EXPECTED_LOGITS, rtol=2e-4, atol=2e-4)

    @slow
    def test_tiny_generation_single(self):
        model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny")
        model.to(torch_device)

        audio_array = self._load_datasamples(1)
        inputs = self.processor_tiny(audio_array, return_tensors="pt")
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_tiny.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_base_generation_single(self):
        model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-base")
        model.to(torch_device)

        audio_array = self._load_datasamples(1)
        inputs = self.processor_base(audio_array, return_tensors="pt")
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_base.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_tiny_generation_batch(self):
        model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny")
        model.to(torch_device)

        audio_array = self._load_datasamples(4)
        inputs = self.processor_tiny(audio_array, return_tensors="pt", padding=True)
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_tiny.batch_decode(generated_ids, skip_special_tokens=True)

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome",
            "Nor is Mr. Quilter's manner less interesting than his matter.",
            "He tells us that at this festive season of the year, with Christmas and Rose beef lo",
            "He has grave doubts whether Sir Frederick Layton's work is really Greek after all,",
        ]
        # fmt: on

        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_base_generation_batch(self):
        model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-base")
        model.to(torch_device)

        audio_array = self._load_datasamples(4)
        inputs = self.processor_base(audio_array, return_tensors="pt", padding=True)
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_base.batch_decode(generated_ids, skip_special_tokens=True)

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome",
            "Nor is Mr. Quilter's manner less interesting than his matter.",
            "He tells us that at this festive season of the year, with Christmas and rose beef lo",
            "He has grave doubts whether Sir Frederick Layton's work is really Greek after all,",
        ]
        # fmt: on

        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)
