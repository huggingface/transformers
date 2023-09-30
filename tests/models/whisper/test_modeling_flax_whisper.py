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
import functools
import inspect
import tempfile
import unittest

import transformers
from transformers import WhisperConfig, is_flax_available
from transformers.testing_utils import is_pt_flax_cross_test, require_flax, slow
from transformers.utils import cached_property
from transformers.utils.import_utils import is_datasets_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor


if is_datasets_available():
    import datasets
    from datasets import load_dataset

if is_flax_available():
    import jax
    import numpy as np
    from flax.core.frozen_dict import unfreeze
    from flax.traverse_util import flatten_dict

    from transformers import (
        FLAX_MODEL_MAPPING,
        FlaxWhisperForAudioClassification,
        FlaxWhisperForConditionalGeneration,
        FlaxWhisperModel,
        WhisperFeatureExtractor,
        WhisperProcessor,
    )
    from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model
    from transformers.models.whisper.modeling_flax_whisper import sinusoidal_embedding_init


@require_flax
class FlaxWhisperModelTester:
    config_cls = WhisperConfig
    config_updates = {}
    hidden_act = "gelu"

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=60,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        d_model=16,
        decoder_attention_heads=4,
        decoder_ffn_dim=16,
        decoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=16,
        encoder_layers=2,
        input_channels=1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=70,
        max_source_positions=30,
        max_target_positions=40,
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
        "decoder_attention_mask": decoder_attention_mask,
    }


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def make_partial_class(full_class, *args, **kwargs):
    partial_class = partialclass(full_class, *args, **kwargs)
    partial_class.__name__ = full_class.__name__
    partial_class.__module__ = full_class.__module__

    return partial_class


@require_flax
class FlaxWhisperModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxWhisperForConditionalGeneration, FlaxWhisperModel) if is_flax_available() else ()
    all_generative_model_classes = (FlaxWhisperForConditionalGeneration,) if is_flax_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = FlaxWhisperModelTester(self)
        _, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.init_shape = (1,) + inputs_dict["input_features"].shape[1:]

        self.all_model_classes = (
            make_partial_class(model_class, input_shape=self.init_shape) for model_class in self.all_model_classes
        )
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

    def check_pt_flax_outputs(self, fx_outputs, pt_outputs, model_class, tol=5e-5, name="outputs", attributes=None):
        # We override with a slightly higher tol value, as test recently became flaky
        super().check_pt_flax_outputs(fx_outputs, pt_outputs, model_class, tol, name, attributes)

    # overwrite because of `input_features`
    @is_pt_flax_cross_test
    def test_save_load_bf16_to_base_pt(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = make_partial_class(FLAX_MODEL_MAPPING[config.__class__], input_shape=self.init_shape)

        for model_class in self.all_model_classes:
            if model_class.__name__ == base_class.__name__:
                continue

            model = model_class(config)
            model.params = model.to_bf16(model.params)
            base_params_from_head = flatten_dict(unfreeze(model.params[model.base_model_prefix]))

            # convert Flax model to PyTorch model
            pt_model_class = getattr(transformers, model_class.__name__[4:])  # Skip the "Flax" at the beginning
            pt_model = pt_model_class(config).eval()
            pt_model = load_flax_weights_in_pytorch_model(pt_model, model.params)

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_model.save_pretrained(tmpdirname)
                base_model = base_class.from_pretrained(tmpdirname, from_pt=True)

                base_params = flatten_dict(unfreeze(base_model.params))

                for key in base_params_from_head.keys():
                    max_diff = (base_params[key] - base_params_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    # overwrite because of `input_features`
    @is_pt_flax_cross_test
    def test_save_load_from_base_pt(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = make_partial_class(FLAX_MODEL_MAPPING[config.__class__], input_shape=self.init_shape)

        for model_class in self.all_model_classes:
            if model_class.__name__ == base_class.__name__:
                continue

            model = base_class(config)
            base_params = flatten_dict(unfreeze(model.params))

            # convert Flax model to PyTorch model
            pt_model_class = getattr(transformers, base_class.__name__[4:])  # Skip the "Flax" at the beginning
            pt_model = pt_model_class(config).eval()
            pt_model = load_flax_weights_in_pytorch_model(pt_model, model.params)

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                # save pt model
                pt_model.save_pretrained(tmpdirname)
                head_model = model_class.from_pretrained(tmpdirname, from_pt=True)

                base_param_from_head = flatten_dict(unfreeze(head_model.params[head_model.base_model_prefix]))

                for key in base_param_from_head.keys():
                    max_diff = (base_params[key] - base_param_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    # overwrite because of `input_features`
    @is_pt_flax_cross_test
    def test_save_load_to_base_pt(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = make_partial_class(FLAX_MODEL_MAPPING[config.__class__], input_shape=self.init_shape)

        for model_class in self.all_model_classes:
            if model_class.__name__ == base_class.__name__:
                continue

            model = model_class(config)
            base_params_from_head = flatten_dict(unfreeze(model.params[model.base_model_prefix]))

            # convert Flax model to PyTorch model
            pt_model_class = getattr(transformers, model_class.__name__[4:])  # Skip the "Flax" at the beginning
            pt_model = pt_model_class(config).eval()
            pt_model = load_flax_weights_in_pytorch_model(pt_model, model.params)

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_model.save_pretrained(tmpdirname)
                base_model = base_class.from_pretrained(tmpdirname, from_pt=True)

                base_params = flatten_dict(unfreeze(base_model.params))

                for key in base_params_from_head.keys():
                    max_diff = (base_params[key] - base_params_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    # overwrite because of `input_features`
    def test_save_load_from_base(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = make_partial_class(FLAX_MODEL_MAPPING[config.__class__], input_shape=self.init_shape)

        for model_class in self.all_model_classes:
            if model_class.__name__ == base_class.__name__:
                continue

            model = base_class(config)
            base_params = flatten_dict(unfreeze(model.params))

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                head_model = model_class.from_pretrained(tmpdirname)

                base_param_from_head = flatten_dict(unfreeze(head_model.params[head_model.base_model_prefix]))

                for key in base_param_from_head.keys():
                    max_diff = (base_params[key] - base_param_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    # overwrite because of `input_features`
    def test_save_load_to_base(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = make_partial_class(FLAX_MODEL_MAPPING[config.__class__], input_shape=self.init_shape)

        for model_class in self.all_model_classes:
            if model_class.__name__ == base_class.__name__:
                continue

            model = model_class(config)
            base_params_from_head = flatten_dict(unfreeze(model.params[model.base_model_prefix]))

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                base_model = base_class.from_pretrained(tmpdirname)

                base_params = flatten_dict(unfreeze(base_model.params))

                for key in base_params_from_head.keys():
                    max_diff = (base_params[key] - base_params_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    def test_encoder_sinusoidal_embed_positions(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            params = model.params
            if model.base_model_prefix in params:
                params = model.params[model.base_model_prefix]

            embeds = params["encoder"]["embed_positions"]["embedding"]
            sinusoids = sinusoidal_embedding_init(None, embeds.shape)
            self.assertTrue(jax.numpy.allclose(embeds, sinusoids))


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

    def test_small_en_logits_librispeech(self):
        model = FlaxWhisperModel.from_pretrained("openai/whisper-small.en", from_pt=True)
        input_speech = self._load_datasamples(1)
        feature_extractor = WhisperFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="np").input_features

        logits = model(
            input_features,
            decoder_input_ids=np.array([model.config.decoder_start_token_id]),
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False,
        )

        logits = logits[0] @ model.params["model"]["decoder"]["embed_tokens"]["embedding"].T

        # fmt: off
        EXPECTED_LOGITS = np.array(
            [
                -3.6784, -7.7211, -9.5070, -11.9286, -7.6489, -9.7026, -5.6188,
                -8.0104, -4.6238, -5.1833, -9.0485, -3.4079, -5.4874, -2.6935,
                -6.3479, -7.3398, -6.9558, -7.6867, -7.4748, -8.3463, -9.9781,
                -10.8389, -10.3105, -11.7201, -9.7261, -7.1590, -5.9272, -12.4509,
                -11.1146, -8.1918
            ]
        )
        # fmt: on
        self.assertTrue(np.allclose(logits[0, 0, :30], EXPECTED_LOGITS, atol=1e-4))

    def test_large_logits_librispeech(self):
        model = FlaxWhisperModel.from_pretrained("openai/whisper-large", from_pt=True)
        input_speech = self._load_datasamples(1)
        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        processed_inputs = processor(
            audio=input_speech, text="This part of the speech", add_special_tokens=False, return_tensors="np"
        )
        input_features = processed_inputs.input_features
        decoder_input_ids = processed_inputs.labels

        logits = model(
            input_features,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False,
        )

        logits = logits[0] @ model.params["model"]["decoder"]["embed_tokens"]["embedding"].T

        # fmt: off
        EXPECTED_LOGITS = np.array(
            [
                2.1382, 0.9381, 4.4671, 3.5589, 2.4022, 3.8576, -0.6521, 2.5472,
                1.8301, 1.9957, 2.3432, 1.4678, 0.5459, 2.2597, 1.5179, 2.5357,
                1.1624, 0.6194, 1.0757, 1.8259, 2.4076, 1.6601, 2.3503, 1.3376,
                1.9891, 1.8635, 3.8931, 5.3699, 4.4772, 3.9184
            ]
        )
        # fmt: on
        self.assertTrue(np.allclose(logits[0, 0, :30], EXPECTED_LOGITS, atol=1e-4))

    def test_tiny_en_generation(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        model.config.decoder_start_token_id = 50257

        input_speech = self._load_datasamples(1)
        input_features = processor.feature_extractor(
            raw_speech=input_speech, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="jax"
        ).input_features

        generated_ids = model.generate(input_features, num_beams=5, max_length=20).sequences
        transcript = processor.tokenizer.decode(generated_ids[0])

        EXPECTED_TRANSCRIPT = (
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle"
            " classes and we are glad to"
        )
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    def test_tiny_generation(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", from_pt=True)

        input_speech = self._load_datasamples(1)
        input_features = processor.feature_extractor(
            raw_speech=input_speech, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="jax"
        ).input_features

        generated_ids = model.generate(input_features, num_beams=5, max_length=20).sequences
        transcript = processor.tokenizer.decode(generated_ids[0])

        EXPECTED_TRANSCRIPT = (
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle"
            " classes and we are glad"
        )
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    def test_large_generation(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-large", from_pt=True)

        input_speech = self._load_datasamples(1)
        input_features = processor.feature_extractor(
            raw_speech=input_speech, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="jax"
        ).input_features

        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

        generated_ids = model.generate(input_features, num_beams=5, max_length=20).sequences
        transcript = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        EXPECTED_TRANSCRIPT = " Mr. Quilter is the apostle of the middle classes and we are glad"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    def test_large_generation_multilingual(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-large", from_pt=True)

        ds = load_dataset("common_voice", "ja", split="test", streaming=True)
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
        input_speech = next(iter(ds))["audio"]["array"]
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="np")

        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")
        generated_ids = model.generate(input_features, do_sample=False, max_length=20).sequences
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = "木村さんに電話を貸してもらいました"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        generated_ids = model.generate(
            input_features,
            do_sample=False,
            max_length=20,
        ).sequences
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = " Kimura-san called me."
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="translate")
        generated_ids = model.generate(input_features, do_sample=False, max_length=20).sequences
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = " I borrowed a phone from Kimura san"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    def test_large_batched_generation(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-large", from_pt=True)

        input_speech = self._load_datasamples(4)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="np").input_features
        generated_ids = model.generate(input_features, max_length=20).sequences

        # fmt: off
        EXPECTED_LOGITS = np.array(
            [
                [50258, 50358, 50363, 2221, 13, 2326, 388, 391, 307, 264, 50244, 295, 264, 2808, 5359, 293, 321, 366, 5404, 281],
                [50258, 50358, 50363, 6966, 307, 2221, 13, 2326, 388, 391, 311, 9060, 1570, 1880, 813, 702, 1871, 13, 50257, 50257],
                [50258, 50358, 50363, 634, 5112, 505, 300, 412, 341, 42729, 3196, 295, 264, 1064, 11, 365, 5272, 293, 12904, 9256],
                [50258, 50358, 50363, 634, 575, 12525, 22618, 1968, 6144, 35617, 20084, 1756, 311, 589, 307, 534, 10281, 934, 439, 11]
            ]
        )
        # fmt: on

        self.assertTrue(np.allclose(generated_ids, EXPECTED_LOGITS))

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            " Mr. Quilter is the apostle of the middle classes and we are glad to",
            " Nor is Mr. Quilter's manner less interesting than his matter.",
            " He tells us that at this festive season of the year, with Christmas and roast beef",
            " He has grave doubts whether Sir Frederick Layton's work is really Greek after all,",
        ]
        # fmt: on

        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)

    def test_tiny_en_batched_generation(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)

        input_speech = self._load_datasamples(4)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="np").input_features
        generated_ids = model.generate(input_features, max_length=20).sequences

        # fmt: off
        EXPECTED_LOGITS = np.array(
            [
                [50257, 50362, 1770, 13, 2264, 346, 353, 318, 262, 46329, 286, 262, 3504, 6097, 11, 290, 356, 389, 9675, 284],
                [50257, 50362, 5414, 318, 1770, 13, 2264, 346, 353, 338, 5642, 1342, 3499, 621, 465, 2300, 13, 50256, 50256, 50256],
                [50257, 50362, 679, 4952, 514, 326, 379, 428, 43856, 1622, 286, 262, 614, 11, 351, 6786, 290, 32595, 12023, 28236],
                [50257, 50362, 679, 468, 12296, 17188, 1771, 7361, 26113, 18881, 1122, 338, 670, 318, 1107, 8312, 706, 477, 290, 460]
            ]

        )
        # fmt: on

        self.assertTrue(np.allclose(generated_ids, EXPECTED_LOGITS))

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            " Mr. Quilter is the apostle of the middle classes, and we are glad to",
            " Nor is Mr. Quilter's manner less interesting than his matter.",
            " He tells us that at this festive season of the year, with Christmas and roast beef looming",
            " He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can",
        ]
        # fmt: on

        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_tiny_timestamp_generation(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

        input_speech = np.concatenate(self._load_datasamples(4))
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="jax").input_features

        generate_fn = jax.jit(functools.partial(model.generate, max_length=448, return_timestamps=True))

        generated_ids = generate_fn(input_features)

        # fmt: off
        EXPECTED_OUTPUT = np.array([50258, 50259, 50359, 50364, 2221, 13, 2326, 388, 391, 307, 264, 50244, 295, 264, 2808, 5359, 11, 293, 321, 366, 5404, 281, 2928, 702, 14943, 13, 50692, 50692, 6966, 307, 2221, 13, 2326, 388, 391, 311, 9060, 1570, 1880, 813, 702, 1871, 13, 50926, 50926, 634, 5112, 505, 300, 412, 341, 42729, 3196, 295, 264, 1064, 11, 365, 5272, 293, 12904, 9256, 450, 10539, 51208, 51208, 949, 505, 11, 14138, 10117, 490, 3936, 293, 1080, 3542, 5160, 881, 26336, 281, 264, 1575, 13, 51552, 51552, 634, 575, 12525, 22618, 1968, 6144, 35617, 7354, 1292, 6, 589, 307, 534, 10281, 934, 439, 11, 293, 51836, 51836, 50257])
        # fmt: on

        self.assertTrue(np.allclose(generated_ids, EXPECTED_OUTPUT))

        EXPECTED_TRANSCRIPT = [
            {
                "text": (
                    " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is"
                    " Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season"
                    " of the year, with Christmas and roast beef looming before us, similarly drawn from eating and"
                    " its results occur most readily to the mind. He has grave doubts whether Sir Frederick Latins'"
                    " work is really Greek after all, and"
                ),
                "offsets": [
                    {
                        "text": (
                            " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
                        ),
                        "timestamp": (0.0, 6.5600000000000005),
                    },
                    {
                        "text": " Nor is Mr. Quilter's manner less interesting than his matter.",
                        "timestamp": (6.5600000000000005, 11.24),
                    },
                    {
                        "text": (
                            " He tells us that at this festive season of the year, with Christmas and roast beef"
                            " looming"
                        ),
                        "timestamp": (11.24, 16.88),
                    },
                    {
                        "text": (
                            " before us, similarly drawn from eating and its results occur most readily to the mind."
                        ),
                        "timestamp": (16.88, 23.76),
                    },
                    {
                        "text": (
                            " He has grave doubts whether Sir Frederick Latins' work is really Greek after all, and"
                        ),
                        "timestamp": (23.76, 29.44),
                    },
                ],
            }
        ]

        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True, output_offsets=True)
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)


class FlaxWhisperEncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=60,
        is_training=True,
        use_labels=True,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        input_channels=1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        max_source_positions=30,
        num_mel_bins=80,
        num_conv_layers=1,
        suppress_tokens=None,
        begin_suppress_tokens=None,
        classifier_proj_size=4,
        num_labels=2,
        is_encoder_decoder=False,
        is_decoder=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.input_channels = input_channels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_mel_bins = num_mel_bins
        self.max_position_embeddings = max_position_embeddings
        self.max_source_positions = max_source_positions
        self.num_conv_layers = num_conv_layers
        self.suppress_tokens = suppress_tokens
        self.begin_suppress_tokens = begin_suppress_tokens
        self.classifier_proj_size = classifier_proj_size
        self.num_labels = num_labels
        self.is_encoder_decoder = is_encoder_decoder
        self.is_decoder = is_decoder

    def get_config(self):
        return WhisperConfig(
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            input_channels=self.input_channels,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            max_source_positions=self.max_source_positions,
            decoder_ffn_dim=self.hidden_size,
            encoder_ffn_dim=self.hidden_size,
            suppress_tokens=self.suppress_tokens,
            begin_suppress_tokens=self.begin_suppress_tokens,
            classifier_proj_size=self.classifier_proj_size,
            num_labels=self.num_labels,
            is_encoder_decoder=self.is_encoder_decoder,
            is_decoder=self.is_decoder,
        )

    def prepare_whisper_encoder_inputs_dict(
        self,
        input_features,
    ):
        return {
            "input_features": input_features,
        }

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.num_mel_bins, self.seq_length])

        config = self.get_config()
        inputs_dict = self.prepare_whisper_encoder_inputs_dict(
            input_features=input_features,
        )
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_subsampled_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        for i in range(self.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    @property
    def encoder_seq_length(self):
        return self.get_subsampled_output_lengths(self.seq_length)


@require_flax
class WhisperEncoderModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxWhisperForAudioClassification,) if is_flax_available() else ()
    is_encoder_decoder = False
    fx_compatible = False
    test_pruning = False
    test_missing_keys = False

    input_name = "input_features"

    def setUp(self):
        self.model_tester = FlaxWhisperEncoderModelTester(self)
        _, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.init_shape = (1,) + inputs_dict["input_features"].shape[1:]

        self.all_model_classes = (
            make_partial_class(model_class, input_shape=self.init_shape) for model_class in self.all_model_classes
        )
        self.config_tester = ConfigTester(self, config_class=WhisperConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    # overwrite because of `input_features`
    def test_jit_compilation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                model = model_class(config)

                @jax.jit
                def model_jitted(input_features, **kwargs):
                    return model(input_features=input_features, **kwargs)

                with self.subTest("JIT Enabled"):
                    jitted_outputs = model_jitted(**prepared_inputs_dict).to_tuple()

                with self.subTest("JIT Disabled"):
                    with jax.disable_jit():
                        outputs = model_jitted(**prepared_inputs_dict).to_tuple()

                self.assertEqual(len(outputs), len(jitted_outputs))
                for jitted_output, output in zip(jitted_outputs, outputs):
                    self.assertEqual(jitted_output.shape, output.shape)

    # overwrite because of `input_features`
    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.__call__)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_features", "attention_mask", "output_attentions"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_inputs_embeds(self):
        pass

    # WhisperEncoder has no inputs_embeds and thus the `get_input_embeddings` fn is not implemented
    def test_model_common_attributes(self):
        pass

    # WhisperEncoder cannot resize token embeddings since it has no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass

    # WhisperEncoder does not have any base model
    def test_save_load_to_base(self):
        pass

    # WhisperEncoder does not have any base model
    def test_save_load_from_base(self):
        pass

    # WhisperEncoder does not have any base model
    @is_pt_flax_cross_test
    def test_save_load_from_base_pt(self):
        pass

    # WhisperEncoder does not have any base model
    @is_pt_flax_cross_test
    def test_save_load_to_base_pt(self):
        pass

    # WhisperEncoder does not have any base model
    @is_pt_flax_cross_test
    def test_save_load_bf16_to_base_pt(self):
        pass
