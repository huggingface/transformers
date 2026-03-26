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

import copy
import math
import unittest

from transformers import AutoProcessor, CohereAsrConfig, CohereAsrForConditionalGeneration, is_torch_available
from transformers.audio_utils import load_audio
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    GenerationTesterMixin,
    ModelTesterMixin,
    floats_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import CohereAsrForConditionalGeneration
    from transformers.models.cohere_asr.modeling_cohere_asr import CohereAsrModel


class CohereAsrModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=256,
        is_training=False,
        encoder_config={
            "model_type": "parakeet_encoder",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "hidden_act": "silu",
            "attention_bias": True,
            "convolution_bias": True,
            "conv_kernel_size": 9,
            "subsampling_factor": 4,
            "subsampling_conv_channels": 8,
            "num_mel_bins": 8,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "dropout": 0.0,
            "dropout_positions": 0.0,
            "layerdrop": 0.0,
            "activation_dropout": 0.0,
            "attention_dropout": 0.0,
            "max_position_embeddings": 5000,
            "scale_input": False,
        },
        decoder_start_token_id=85,
        bos_token_id=98,
        eos_token_id=98,
        pad_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.encoder_config = encoder_config
        self.decoder_start_token_id = decoder_start_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # Decoder defaults
        self.vocab_size = 147
        self.hidden_size = 16
        self.intermediate_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 2
        self.num_key_value_heads = 2
        self.head_dim = 8

        # Derived from encoder_config for test assertions
        self.num_mel_bins = encoder_config["num_mel_bins"]
        self.encoder_hidden_size = encoder_config["hidden_size"]
        self.encoder_num_hidden_layers = encoder_config["num_hidden_layers"]
        self.encoder_num_attention_heads = encoder_config["num_attention_heads"]
        self.encoder_seq_length = self.get_encoder_output_length(seq_length)
        self.decoder_seq_length = 1
        self.decoder_key_length = 1
        self.key_length = self.encoder_seq_length

    def get_encoder_output_length(self, input_length):
        """Compute the encoder output length after subsampling convolutions."""
        num_layers = int(math.log2(self.encoder_config["subsampling_factor"]))
        kernel_size = self.encoder_config["subsampling_conv_kernel_size"]
        stride = self.encoder_config["subsampling_conv_stride"]
        add_pad = (kernel_size - 1) // 2 * 2 - kernel_size
        length = input_length
        for _ in range(num_layers):
            length = int((length + add_pad) / stride) + 1
        return length

    def get_config(self):
        return CohereAsrConfig(
            encoder_config=self.encoder_config,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            hidden_act="relu",
            attention_bias=True,
            attention_dropout=0.0,
            decoder_start_token_id=self.decoder_start_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        decoder_input_ids = torch.tensor(self.batch_size * [[self.decoder_start_token_id]], device=torch_device)
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id)
        config = self.get_config()
        return config, input_features, attention_mask, decoder_input_ids, decoder_attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask, decoder_input_ids, decoder_attention_mask = (
            self.prepare_config_and_inputs()
        )
        inputs_dict = {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict


@require_torch
class CohereAsrModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (CohereAsrModel, CohereAsrForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (CohereAsrForConditionalGeneration,)
    pipeline_model_mapping = (
        {
            "automatic-speech-recognition": CohereAsrForConditionalGeneration,
            "feature-extraction": CohereAsrModel,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True

    # CohereAsr's pos_emb layer is large relative to total model size
    model_split_percents = [0.5, 0.9, 0.95]

    def setUp(self):
        self.model_tester = CohereAsrModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CohereAsrConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        # proj_out conversion only applies to ForConditionalGeneration, not the base model
        try:
            self.all_model_classes = (CohereAsrForConditionalGeneration,) if is_torch_available() else ()
            super().test_reverse_loading_mapping(check_keys_were_modified)
        finally:
            self.all_model_classes = (
                (CohereAsrModel, CohereAsrForConditionalGeneration) if is_torch_available() else ()
            )

    # Copied from tests.models.moonshine_streaming.test_modeling_moonshine_streaming.MoonshineStreamingModelTest.test_resize_tokens_embeddings
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

    # Copied from tests.models.moonshine_streaming.test_modeling_moonshine_streaming.MoonshineStreamingModelTest.test_resize_embeddings_untied
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

    @unittest.skip(reason="Not known - aborted for now, not super important")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="FIXME: likely intended because we need input ids but to double-check")
    def test_generate_without_input_ids(self):
        pass


# TODO: remove revision
@require_torch
class CohereAsrIntegrationTest(unittest.TestCase):
    checkpoint_name = "CohereLabs/cohere-transcribe-03-2026"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name, revision="refs/pr/6")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_shortform_english(self):
        """
        reproducer: https://gist.github.com/eustlb/cfcea58b4ffabfd45b4b6fce5ab283ed
        """
        audio = load_audio(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
            sampling_rate=16000,
        )
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            language="en",
        )
        model = CohereAsrForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, revision="refs/pr/6"
        )
        inputs.to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            " Yesterday it was thirty-five degrees in Barcelona, but today the temperature will go down to minus twenty degrees."
        ]
        self.assertEqual(text, EXPECTED_OUTPUT)

    @slow
    def test_shortform_english_no_punctuation(self):
        """
        reproducer: https://gist.github.com/eustlb/cfcea58b4ffabfd45b4b6fce5ab283ed
        """
        audio = load_audio(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
            sampling_rate=16000,
        )
        inputs_pnc = self.processor(audio, sampling_rate=16000, return_tensors="pt", language="en", punctuation=True)
        inputs_nopnc = self.processor(
            audio, sampling_rate=16000, return_tensors="pt", language="en", punctuation=False
        )

        model = CohereAsrForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, revision="refs/pr/6"
        )
        inputs_pnc.to(model.device, dtype=model.dtype)
        inputs_nopnc.to(model.device, dtype=model.dtype)

        outputs_pnc = model.generate(**inputs_pnc, max_new_tokens=256)
        outputs_nopnc = model.generate(**inputs_nopnc, max_new_tokens=256)

        text_pnc = self.processor.decode(outputs_pnc, skip_special_tokens=True)
        text_nopnc = self.processor.decode(outputs_nopnc, skip_special_tokens=True)

        EXPECTED_OUTPUT_PNC = [
            " Yesterday it was thirty-five degrees in Barcelona, but today the temperature will go down to minus twenty degrees."
        ]
        EXPECTED_OUTPUT_NOPNC = [
            " yesterday it was thirty-five degrees in barcelona but today the temperature will go down to minus twenty degrees"
        ]
        self.assertEqual(text_pnc, EXPECTED_OUTPUT_PNC)
        self.assertEqual(text_nopnc, EXPECTED_OUTPUT_NOPNC)

    @slow
    def test_longform_english(self):
        """
        reproducer: https://gist.github.com/eustlb/cfcea58b4ffabfd45b4b6fce5ab283ed
        """
        audio = load_audio(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3",
            sampling_rate=16000,
        )
        inputs = self.processor(audio=audio, return_tensors="pt", language="en", sampling_rate=16000)
        audio_chunk_index = inputs.get("audio_chunk_index")
        model = CohereAsrForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, revision="refs/pr/6"
        )
        inputs.to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(
            outputs, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="en"
        )

        # fmt: off
        EXPECTED_OUTPUT = [
            " This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, all these conversations are what have kept me honest, kept me inspired, and kept me going. Every day I learned from you. You made me a better president and you made me a better man. Over the course of these eight years, I've seen the goodness, the resilience, and the hope of the American."
        ]
        # fmt: on
        self.assertEqual(text, EXPECTED_OUTPUT)

    @slow
    def test_batched_mixed_lengths(self):
        """
        reproducer: https://gist.github.com/eustlb/cfcea58b4ffabfd45b4b6fce5ab283ed
        """
        audio_short = load_audio(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
            sampling_rate=16000,
        )
        audio_long = load_audio(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3",
            sampling_rate=16000,
        )
        inputs = self.processor([audio_short, audio_long], sampling_rate=16000, return_tensors="pt", language="en")
        audio_chunk_index = inputs.get("audio_chunk_index")
        model = CohereAsrForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, revision="refs/pr/6"
        )
        inputs.to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(
            outputs, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="en"
        )

        # fmt: off
        EXPECTED_OUTPUT = [
            " Yesterday it was thirty-five degrees in Barcelona, but today the temperature will go down to minus twenty degrees.",
            " This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, all these conversations are what have kept me honest, kept me inspired, and kept me going. Every day I learned from you. You made me a better president and you made me a better man. Over the course of these eight years, I've seen the goodness, the resilience, and the hope of the American.",
        ]
        # fmt: on
        self.assertEqual(text, EXPECTED_OUTPUT)

    @slow
    def test_non_english_with_punctuation(self):
        """
        reproducer: https://gist.github.com/eustlb/cfcea58b4ffabfd45b4b6fce5ab283ed
        """
        audio = load_audio(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/fleur_es_sample.wav",
            sampling_rate=16000,
        )
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", language="es", punctuation=True)
        model = CohereAsrForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, revision="refs/pr/6"
        )
        inputs.to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [" Esto parece tener sentido ya que en la Tierra no se percibe su movimiento, ¿cierto?"]
        self.assertEqual(text, EXPECTED_OUTPUT)
