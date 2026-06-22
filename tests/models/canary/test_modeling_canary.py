# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Canary model."""

import copy
import math
import unittest

from transformers import CanaryConfig, ParakeetEncoderConfig, is_torch_available
from transformers.testing_utils import is_flaky, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import CanaryForConditionalGeneration, CanaryModel


class CanaryModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=80,
        is_training=False,
        use_labels=False,
        num_mel_bins=80,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        subsampling_factor=8,
        subsampling_conv_channels=16,
        decoder_seq_length=4,
        vocab_size=99,
        max_target_positions=40,
        decoder_start_token_id=7,
        pad_token_id=2,
        bos_token_id=4,
        eos_token_id=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.decoder_seq_length = decoder_seq_length
        self.decoder_key_length = decoder_seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_mel_bins = num_mel_bins
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.vocab_size = vocab_size
        self.max_target_positions = max_target_positions
        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def get_config(self):
        encoder_config = ParakeetEncoderConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            num_mel_bins=self.num_mel_bins,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            scale_input=False,
        )
        return CanaryConfig(
            encoder_config=encoder_config,
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            decoder_layers=self.num_hidden_layers,
            decoder_attention_heads=self.num_attention_heads,
            decoder_ffn_dim=self.intermediate_size,
            max_target_positions=self.max_target_positions,
            decoder_start_token_id=self.decoder_start_token_id,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        # `seq_length` is in mel frames; keep it a multiple of `subsampling_factor` so 8x subsampling does not collapse it.
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins], scale=1.0)
        attention_mask = torch.ones([self.batch_size, self.seq_length], dtype=torch.long, device=torch_device)
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id)
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict

    def get_subsampled_output_lengths(self, input_lengths):
        """Computes the FastConformer subsampled length, used by the generation test mixin for encoder shapes."""
        kernel_size, stride = 3, 2
        padding = (kernel_size - 1) // 2 * 2 - kernel_size
        for _ in range(int(math.log2(self.subsampling_factor))):
            input_lengths = (input_lengths + padding) // stride + 1
        return input_lengths


@require_torch
class CanaryModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (CanaryModel, CanaryForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (CanaryForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "automatic-speech-recognition": CanaryForConditionalGeneration,
            "feature-extraction": CanaryModel,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    test_pruning = False
    test_resize_embeddings = True
    test_headmasking = False

    def setUp(self):
        self.model_tester = CanaryModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CanaryConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    # Overridden: the FastConformer encoder subsamples the input, so encoder shapes use the subsampled length (like Whisper).
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

            subsampled_seq_length = self.model_tester.get_subsampled_output_lengths(self.model_tester.seq_length)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [subsampled_seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states
                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [self.model_tester.decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    # Overridden for the same subsampling reason as `test_hidden_states_output` (mirrors `WhisperModelTest`).
    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        # force eager attention to support output attentions
        config._attn_implementation = "eager"

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

            subsampled_encoder_seq_length = self.model_tester.get_subsampled_output_lengths(encoder_seq_length)
            subsampled_encoder_key_length = self.model_tester.get_subsampled_output_lengths(encoder_key_length)

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
                [self.model_tester.num_attention_heads, decoder_seq_length, subsampled_encoder_key_length],
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

    # Overridden because Canary is an audio model: it takes `input_features` + `decoder_input_ids`, not `input_ids` (like Whisper).
    def test_resize_tokens_embeddings(self):
        original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is False")

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)
            if self.model_tester.is_training is False:
                model.eval()

            # Retrieve the embeddings and clone theme
            model_vocab_size = config.vocab_size
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

    # Overridden for the same audio-model reason as `test_resize_tokens_embeddings` (mirrors `WhisperModelTest`).
    def test_resize_embeddings_untied(self):
        original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
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
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

    @unittest.skip(
        reason="Canary is an encoder-decoder ASR model that requires audio features and cannot generate from input ids only."
    )
    def test_generate_without_input_ids(self):
        pass

    @is_flaky(description="Large difference with A10. Still flaky after setting larger tolerance")
    def test_generate_continue_from_past_key_values(self):
        super().test_generate_continue_from_past_key_values()


@require_torch
@slow
class CanaryIntegrationTest(unittest.TestCase):
    checkpoint = "harshaljanjani/canary-1b-v2-hf"

    def _load(self):
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.checkpoint)
        model = CanaryForConditionalGeneration.from_pretrained(self.checkpoint).to(torch_device).eval()
        return processor, model

    def _sample(self, processor):
        from datasets import Audio, load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
        return ds[0]["audio"]["array"]

    def test_transcription_en(self):
        processor, model = self._load()
        inputs = self._sample(processor)
        features = processor(inputs, source_lang="en", target_lang="en", pnc=True, return_tensors="pt").to(
            torch_device
        )
        generated = model.generate(**features, max_new_tokens=128)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        self.assertEqual(
            text.strip(),
            "mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
        )

    def test_translation_en_to_de(self):
        processor, model = self._load()
        inputs = self._sample(processor)
        features = processor(inputs, source_lang="en", target_lang="de", pnc=True, return_tensors="pt").to(
            torch_device
        )
        generated = model.generate(**features, max_new_tokens=128)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        self.assertEqual(
            text.strip(),
            "Mister Quilter ist der Apostel der Mittelschicht, und wir freuen uns, sein Evangelium willkommen zu heißen.",
        )
