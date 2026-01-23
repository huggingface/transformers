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


import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceAcousticTokenizerModel,
)
from transformers.audio_utils import load_audio_librosa
from transformers.testing_utils import cleanup, is_torch_available, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch


@require_torch
class VibeVoiceAcousticTokenizerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        channels=1,
        hidden_size=32,
        kernel_size=3,
        n_filters=4,
        downsampling_ratios=[2],
        depths=[1, 1],
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.channels = channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.downsampling_ratios = downsampling_ratios
        self.depths = depths

    def prepare_config_and_inputs(self):
        audio = floats_tensor([self.batch_size, self.channels, self.hidden_size], scale=1.0)
        config = self.get_config()
        # disable sampling for deterministic tests
        inputs_dict = {"audio": audio, "sample": False}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        audio = floats_tensor([self.batch_size, self.channels, self.hidden_size], scale=1.0)
        config = self.get_config()
        # disable sampling for deterministic tests
        inputs_dict = {"audio": audio, "sample": False}

        return config, inputs_dict

    def get_config(self):
        return VibeVoiceAcousticTokenizerConfig(
            channels=self.channels,
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
            n_filters=self.n_filters,
            downsampling_ratios=self.downsampling_ratios,
            depths=self.depths,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = VibeVoiceAcousticTokenizerModel(config=config).to(torch_device).eval()

        audio = inputs_dict["audio"]
        result = model(audio)

        # Calculate expected sequence length after downsampling
        expected_seq_len = self.hidden_size // np.prod(self.downsampling_ratios)
        self.parent.assertEqual(result.latents.shape, (self.batch_size, expected_seq_len, self.hidden_size))
        # Acoustic tokenizer should reconstruct audio
        self.parent.assertEqual(result.audio.shape, (self.batch_size, self.channels, self.hidden_size))


@require_torch
class VibeVoiceAcousticTokenizerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (VibeVoiceAcousticTokenizerModel,) if is_torch_available() else ()
    is_encoder_decoder = False
    test_resize_embeddings = False
    test_head_masking = False
    test_pruning = False
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = VibeVoiceAcousticTokenizerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=VibeVoiceAcousticTokenizerConfig,
            common_properties=[],
            has_text_modality=False,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["audio", "padding_cache", "use_cache", "sample"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @unittest.skip("VibeVoiceAcousticTokenizerModel does not have `inputs_embeds` logic")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("VibeVoiceAcousticTokenizerModel does not have `inputs_embeds` logic")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("VibeVoiceAcousticTokenizerModel does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("VibeVoiceAcousticTokenizerModel does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip("VibeVoiceAcousticTokenizerModel does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="From CI 'UnboundLocalError: local variable output referenced before assignment'")
    def test_model_parallelism(self):
        pass

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            out_1 = first.cpu().numpy()
            out_2 = second.cpu().numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class)).latents
                second = model(**self._prepare_for_class(inputs_dict, model_class)).latents

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (list, tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

    def test_encode_method(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceAcousticTokenizerModel(config=config).to(torch_device).eval()

        audio = inputs_dict["audio"]
        with torch.no_grad():
            output = model.encode(audio)

        self.assertIsNotNone(output.latents)
        expected_seq_len = self.model_tester.hidden_size // np.prod(self.model_tester.downsampling_ratios)
        self.assertEqual(
            output.latents.shape, (self.model_tester.batch_size, expected_seq_len, self.model_tester.hidden_size)
        )

    def test_decode_method(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceAcousticTokenizerModel(config=config).to(torch_device).eval()

        # First encode to get latents
        audio = inputs_dict["audio"]
        with torch.no_grad():
            encode_output = model.encode(audio)
            decode_output = model.decode(encode_output.latents)

        self.assertIsNotNone(decode_output.audio)
        # Decoder should reconstruct to original audio shape
        self.assertEqual(
            decode_output.audio.shape,
            (self.model_tester.batch_size, self.model_tester.channels, self.model_tester.hidden_size),
        )

    def test_use_cache(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceAcousticTokenizerModel(config=config).to(torch_device).eval()

        audio = inputs_dict["audio"]
        with torch.no_grad():
            output = model(audio, use_cache=True)

        self.assertIsNotNone(output.padding_cache)
        self.assertIsNotNone(output.latents)
        self.assertIsNotNone(output.audio)


class VibeVoiceAcousticTokenizerIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_checkpoint = "bezzam/VibeVoice-AcousticTokenizer"
        self.sampling_rate = 24000

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch
    def test_batch_integration(self):
        """
        Reproducer which generates JSON of expected outputs:
        https://gist.github.com/ebezzam/507dfd544e0a0f12402966503cbc73e6#file-reproducer_tokenizer-py
        NOTE (ebezzam): had to compute expected outputs on CI runners for passing tests
        """
        dtype = torch.bfloat16

        # Load expected outputs
        RESULTS_PATH = (
            Path(__file__).parent.parent.parent / "fixtures/vibevoice/expected_acoustic_tokenizer_results.json"
        )
        with open(RESULTS_PATH, "r") as f:
            expected_results = json.load(f)
        expected_encoder = torch.tensor(expected_results["encoder"]).to(dtype)
        expected_decoder = torch.tensor(expected_results["decoder"]).to(dtype)

        # Prepare inputs
        audio_paths = [
            "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Carter_man.wav",
            "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Frank_man.wav",
        ]
        audio_arrays = [load_audio_librosa(path, sampling_rate=self.sampling_rate) for path in audio_paths]
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_checkpoint)

        # apply model and compare
        model = AutoModel.from_pretrained(
            self.model_checkpoint,
            dtype=dtype,
            device_map=torch_device,
        ).eval()
        processed_audio = feature_extractor(audio_arrays, return_tensors="pt", sampling_rate=self.sampling_rate).to(
            torch_device, dtype=dtype
        )
        with torch.no_grad():
            encoder_out = model.encode(processed_audio["input_values"]).latents
            acoustic_decoder_out = model.decode(encoder_out).audio
        encoder_out_flat = encoder_out.reshape(encoder_out.shape[0], -1)
        encoder_out = encoder_out_flat[..., : expected_encoder.shape[-1]].cpu()
        decoder_out = acoustic_decoder_out[..., : expected_decoder.shape[-1]].cpu()
        torch.testing.assert_close(encoder_out, expected_encoder, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(decoder_out, expected_decoder, rtol=1e-6, atol=1e-6)
