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
"""Testing suite for the PyTorch Moonshine model."""

import copy
import unittest

from transformers import MoonshineConfig, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
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
        seq_length=60,
        is_training=True,
        use_labels=False,
        vocab_size=147,
        hidden_size=8,
        conv1_kernel_size=7,
        conv1_stride=3,
        conv2_kernel_size=7,
        conv2_stride=3,
        conv3_kernel_size=3,
        conv3_stride=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        min_rotary_ndims=4,
        encoder_hidden_act="gelu",
        decoder_hidden_act="silu",
        attention_dropout=0.1,
        intermediate_size=None,
        ff_mult=2,
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
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_stride = conv1_stride
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_stride = conv2_stride
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_stride = conv3_stride
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.min_rotary_ndims = min_rotary_ndims
        self.encoder_hidden_act = encoder_hidden_act
        self.decoder_hidden_act = decoder_hidden_act
        self.attention_dropout = attention_dropout
        self.decoder_start_token_id = decoder_start_token_id
        self.intermediate_size = intermediate_size
        self.ff_mult = ff_mult
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
            num_hidden_layers=self.num_hidden_layers,
            conv1_kernel_size=self.conv1_kernel_size,
            conv1_stride=self.conv1_stride,
            conv2_kernel_size=self.conv2_kernel_size,
            conv2_stride=self.conv2_stride,
            conv3_kernel_size=self.conv3_kernel_size,
            conv3_stride=self.conv3_stride,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            min_rotary_ndims=self.min_rotary_ndims,
            encoder_hidden_act=self.encoder_hidden_act,
            decoder_hidden_act=self.decoder_hidden_act,
            decoder_start_token_id=self.decoder_start_token_id,
            attention_dropout=self.attention_dropout,
            ff_mult=self.ff_mult,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    def create_and_check_model(self, config, input_values, attention_mask):
        model = MoonshineModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_batch_inference(self, config, input_values, *args):
        # test does not pass for models making use of `group_norm`
        # check: https://github.com/pytorch/fairseq/issues/3227
        model = MoonshineModel(config=config)
        model.to(torch_device)
        model.eval()

        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.bool)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0.0

        batch_outputs = model(input_values, attention_mask=attention_mask).last_hidden_state

        for i in range(input_values.shape[0]):
            input_slice = input_values[i : i + 1, : input_lengths[i]]
            output = model(input_slice).last_hidden_state

            batch_output = batch_outputs[i : i + 1, : output.shape[1]]
            self.parent.assertTrue(torch.allclose(output, batch_output, atol=1e-3))

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
class MoonshineModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MoonshineModel, MoonshineForConditionalGeneration) if is_torch_available() else ()
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
            model = model_class(config)
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

    # training is not supported yet
    @unittest.skip(reason="Training is not supported yet")
    def test_training(self):
        pass

    @unittest.skip(reason="Training is not supported yet")
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


@require_torch
class MoonshineModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.processor_tiny = AutoProcessor.from_pretrained("eustlb/moonshine-tiny")
        self.processor_base = AutoProcessor.from_pretrained("eustlb/moonshine-base")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @slow
    def test_tiny_logits_single(self):
        model = MoonshineForConditionalGeneration.from_pretrained("eustlb/moonshine-tiny")
        model.to(torch_device)

        inputs = self.processor_tiny(self._load_datasamples(1), return_tensors="pt")
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            -5.9775,  4.0908, 11.3396, -7.3772, -7.7979, -8.2989, -7.7634, -8.0795, -8.5730, -8.3415,
            -7.8777, -7.8082, -8.1875, -7.9751, -7.8597, -8.1751, -8.1304, -7.5688, -7.9222, -7.8248,
            -8.5372, -7.8276, -7.8369, -7.4459, -8.2137, -7.8129, -7.4717, -7.2522, -8.2155, -7.7587,
        ])
        # fmt: on
        self.assertTrue(torch.allclose(outputs.logits[0][0, :30].cpu(), EXPECTED_LOGITS, atol=1e-4))

    @slow
    def test_base_logits_single(self):
        model = MoonshineForConditionalGeneration.from_pretrained("eustlb/moonshine-base")
        model.to(torch_device)

        inputs = self.processor_base(self._load_datasamples(1), return_tensors="pt")
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            -6.7340,  1.9483,  5.2449, -8.0277, -7.9167, -7.8956, -7.9649, -7.9348, -8.1312, -8.0616,
            -8.1070, -7.7696, -7.8809, -7.9451, -8.1013, -7.8177, -7.8598, -7.8257, -7.8729, -7.9657,
            -7.9310, -8.1024, -7.8698, -7.8231, -8.0752, -7.9764, -7.8127, -8.0536, -7.9492, -7.9289,
        ])
        # fmt: on
        self.assertTrue(torch.allclose(outputs.logits[0][0, :30].cpu(), EXPECTED_LOGITS, atol=1e-4))

    @slow
    def test_tiny_logits_batch(self):
        model = MoonshineForConditionalGeneration.from_pretrained("eustlb/moonshine-tiny")
        model.to(torch_device)

        inputs = self.processor_tiny(self._load_datasamples(4), return_tensors="pt", padding=True)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)
        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            [-4.9763,  4.5247, 10.2098, -7.4247, -7.7815, -8.3471, -7.7532, -8.0495, -8.5072, -8.2964],
            [-3.1452, -1.7097, 10.9730, -7.3581, -7.8165, -7.5148, -7.0205, -7.4499, -7.7192, -7.3988],
            [-7.3976,  2.7249,  6.5921, -7.2217, -7.4929, -7.1929, -7.0791, -7.5758, -7.2907, -7.2343],
            [-7.1467,  3.4882,  7.0442, -6.0044, -6.3131, -6.4540, -6.2300, -6.4451, -6.2651, -6.2361]
        ])
        # fmt: on
        self.assertTrue(torch.allclose(outputs.logits[0][:, :10].cpu(), EXPECTED_LOGITS, atol=1e-4))

    @slow
    def test_base_logits_batch(self):
        model = MoonshineForConditionalGeneration.from_pretrained("eustlb/moonshine-base")
        model.to(torch_device)

        inputs = self.processor_base(self._load_datasamples(4), return_tensors="pt", padding=True)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            [-7.7288,  1.4636,  5.2273, -7.7310, -7.6249, -7.6009, -7.6786, -7.6438, -7.8450, -7.7546],
            [-6.2161, -0.5891,  7.9489, -7.0693, -6.9996, -6.9980, -7.0952, -7.0830, -7.1685, -7.0136],
            [-7.3186,  3.1192,  3.8938, -5.7208, -5.8429, -5.7610, -5.9997, -5.8213, -5.8616, -5.8720],
            [-9.5488,  1.0147,  4.1174, -5.9972, -6.0616, -6.0331, -6.2105, -6.0320, -6.0791, -6.0875]
        ])

        # fmt: on
        self.assertTrue(torch.allclose(outputs.logits[0][:, :10].cpu(), EXPECTED_LOGITS, atol=1e-4))

    @slow
    def test_tiny_generation_single(self):
        model = MoonshineForConditionalGeneration.from_pretrained("eustlb/moonshine-tiny")
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
        model = MoonshineForConditionalGeneration.from_pretrained("eustlb/moonshine-base")
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
        model = MoonshineForConditionalGeneration.from_pretrained("eustlb/moonshine-tiny")
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
        model = MoonshineForConditionalGeneration.from_pretrained("eustlb/moonshine-base")
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
