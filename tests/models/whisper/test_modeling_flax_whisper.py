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


import inspect
import unittest

from datasets import load_dataset

from transformers import WhisperConfig, is_flax_available
from transformers.testing_utils import require_flax, slow
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor


if is_flax_available():
    import numpy as np

    import jax
    from transformers import (
        FlaxWhisperForConditionalGeneration,
        FlaxWhisperModel,
        WhisperFeatureExtractor,
        WhisperProcessor,
    )


@require_flax
class FlaxWhisperModelTester:
    config_cls = WhisperConfig
    config_updates = {}
    hidden_act = "gelu"

    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=3000,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        d_model=384,
        decoder_attention_heads=6,
        decoder_ffn_dim=1536,
        decoder_layers=4,
        encoder_attention_heads=6,
        encoder_ffn_dim=1536,
        encoder_layers=4,
        input_channels=1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        max_source_positions=1500,
        max_target_positions=448,
        bos_token_id=98,
        eos_token_id=98,
        pad_token_id=0,
        num_mel_bins=80,
        decoder_start_token_id=85,
        num_conv_layers=1,
        suppress_tokens=None,
        begin_suppress_tokens=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model
        self.num_hidden_layers = encoder_layers
        self.num_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_seq_length = seq_length // 2
        self.decoder_seq_length = 1
        self.input_channels = input_channels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_mel_bins = num_mel_bins
        self.max_position_embeddings = max_position_embeddings
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.num_conv_layers = num_conv_layers
        self.suppress_tokens = suppress_tokens
        self.begin_suppress_tokens = begin_suppress_tokens

    def prepare_config_and_inputs_for_common(self):
        input_features = floats_tensor([self.batch_size, self.num_mel_bins, self.seq_length], self.vocab_size)

        decoder_input_ids = np.array(self.batch_size * [[self.decoder_start_token_id]])

        config = WhisperConfig(
            vocab_size=self.vocab_size,
            num_mel_bins=self.num_mel_bins,
            decoder_start_token_id=self.decoder_start_token_id,
            is_encoder_decoder=True,
            activation_function=self.hidden_act,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_source_positions=self.max_source_positions,
            max_target_positions=self.max_target_positions,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            tie_word_embeddings=True,
            d_model=self.d_model,
            decoder_attention_heads=self.decoder_attention_heads,
            decoder_ffn_dim=self.decoder_ffn_dim,
            decoder_layers=self.decoder_layers,
            encoder_attention_heads=self.encoder_attention_heads,
            encoder_ffn_dim=self.encoder_ffn_dim,
            encoder_layers=self.encoder_layers,
            suppress_tokens=self.suppress_tokens,
            begin_suppress_tokens=self.begin_suppress_tokens,
        )
        inputs_dict = prepare_whisper_inputs_dict(config, input_features, decoder_input_ids)
        return config, inputs_dict


def prepare_whisper_inputs_dict(
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
        "input_features": input_ids,
        "decoder_input_ids": decoder_input_ids,
    }


@require_flax
class FlaxWhisperModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            FlaxWhisperForConditionalGeneration,
            FlaxWhisperModel,
        )
        if is_flax_available()
        else ()
    )
    all_generative_model_classes = (FlaxWhisperForConditionalGeneration,) if is_flax_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = FlaxWhisperModelTester(self)
        self.config_tester = ConfigTester(self, config_class=WhisperConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    # overwrite because of `input_features`
    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.__call__)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_features", "decoder_input_ids"]
            self.assertListEqual(arg_names[:2], expected_arg_names)

    # overwrite because of `input_features`
    def test_jit_compilation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                model = model_class(config)

                @jax.jit
                def model_jitted(input_features, decoder_input_ids, **kwargs):
                    return model(input_features=input_features, decoder_input_ids=decoder_input_ids, **kwargs)

                with self.subTest("JIT Enabled"):
                    jitted_outputs = model_jitted(**prepared_inputs_dict).to_tuple()

                with self.subTest("JIT Disabled"):
                    with jax.disable_jit():
                        outputs = model_jitted(**prepared_inputs_dict).to_tuple()

                self.assertEqual(len(outputs), len(jitted_outputs))
                for jitted_output, output in zip(jitted_outputs, outputs):
                    self.assertEqual(jitted_output.shape, output.shape)


def _assert_tensors_equal(a, b, atol=1e-12, prefix=""):
    """If tensors not close, or a and b arent both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if _assert_tensors_equal(a, b, atol=atol):
            return True
        raise
    except Exception:
        if len(prefix) > 0:
            prefix = f"{prefix}: "
        raise AssertionError(f"{prefix}{a} != {b}")


def _long_tensor(tok_lst):
    return np.array(tok_lst, dtype=np.int32)


TOLERANCE = 1e-4


@slow
@require_flax
class FlaxWhisperModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return WhisperProcessor.from_pretrained("openai/whisper-base")

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_tiny_logits_librispeech(self):
        model = FlaxWhisperModel.from_pretrained("openai/whisper-tiny", from_pt=True)
        input_speech = self._load_datasamples(1)
        feature_extractor = WhisperFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="np").input_features

        logits = model(
            input_features,
            decoder_input_ids=np.array([[50258, 50259, 50359]]),
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False,
        )

        # fmt: off
        EXPECTED_LOGITS = np.array(
            [
                2.9892, -6.7607, 5.7348, 3.6096, 0.2152, -5.7321, 4.8855, -1.6407,
                0.2823, -1.5718, 10.4269, 3.4427, 0.0219, -8.0612, 3.4784, 8.4246,
                4.0575, -2.2864, 11.1084, 0.9963, 0.9884, -8.5154, -3.5469, -9.3713,
                0.9786, 3.5435, 7.4850, -5.2579, -1.4366, 10.4841
            ]
        )
        # fmt: on
        self.assertTrue(np.allclose(logits[0][0, 0, :30], EXPECTED_LOGITS, atol=1e-4))
