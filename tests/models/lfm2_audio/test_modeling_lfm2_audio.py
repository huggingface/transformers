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
"""Tests for LFM2-Audio."""

import hashlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

from parameterized import parameterized

from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    GenerationConfig,
    Lfm2AudioConfig,
    Lfm2AudioDetokenizer,
    Lfm2AudioForConditionalGeneration,
    Lfm2AudioModel,
    Lfm2Config,
)
from transformers.testing_utils import (
    cleanup,
    require_librosa,
    require_torch,
    require_torch_gpu,
    require_torchaudio,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


class Lfm2AudioModelTester:
    def __init__(self, parent, batch_size=2, sequence_length=8):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = sequence_length
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.vocab_size = 32
        self.is_training = True

    def get_config(self):
        return Lfm2AudioConfig(
            codebooks=2,
            audio_vocab_size=17,
            audio_eos_token_id=16,
            audio_token_id=6,
            audio_start_token_id=4,
            text_end_token_id=5,
            bos_token_id=1,
            eos_token_id=3,
            pad_token_id=0,
            encoder={
                "num_mel_bins": 8,
                "num_hidden_layers": 1,
                "hidden_size": 16,
                "num_attention_heads": 4,
                "subsampling_conv_channels": 8,
                "dropout": 0.0,
                "intermediate_size": 64,
                "layerdrop": 0.0,
                "scale_input": False,
                "dropout_positions": 0.0,
                "attention_dropout": 0.0,
            },
            lfm={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": 64,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": 2,
                "layer_types": ["full_attention", "conv"],
                "max_position_embeddings": 64,
            },
            depthformer={
                "layers": 2,
                "dim": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": 2,
                "intermediate_size": 64,
            },
        )

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids[input_ids == config.audio_token_id] = 2
        attention_mask = input_ids.new_ones(input_ids.shape)
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class Lfm2AudioModelTest(ModelTesterMixin, unittest.TestCase):
    _is_composite = True
    # The generic training inputs are text-only; multimodal gradients are checked separately.
    test_all_params_have_gradient = False
    # Audio-frame generation has a custom output and currently supports only one sample.
    all_generative_model_classes = ()

    all_model_classes = (Lfm2AudioModel, Lfm2AudioForConditionalGeneration)

    def setUp(self):
        self.model_tester = Lfm2AudioModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Lfm2AudioConfig, has_text_modality=False)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels and model_class is Lfm2AudioForConditionalGeneration:
            inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    def _audio_features_prepare_config_and_inputs(self):
        config = self.model_tester.get_config()
        return config, {
            "input_features": floats_tensor([2, 32, config.encoder.num_mel_bins]),
            "input_features_attention_mask": torch.ones(2, 32, device=torch_device, dtype=torch.long),
        }

    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        return self.model_tester.get_config().encoder.num_hidden_layers

    def test_attention_outputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            outputs = model(**inputs, output_attentions=True)
            self.assertEqual(len(outputs.attentions), config.lfm.layer_types.count("full_attention"))
            self.assertEqual(outputs.attentions[0].shape[-2:], (self.model_tester.seq_length,) * 2)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_text_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            outputs = model(**inputs)
            expected_shape = (self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size)
            if model_class is Lfm2AudioForConditionalGeneration:
                expected_shape = (self.model_tester.batch_size, self.model_tester.seq_length, config.vocab_size)
            self.assertEqual(outputs[0].shape, expected_shape)

    def test_save_and_load(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                expected = model(**inputs)[0]
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                reloaded = model_class.from_pretrained(directory).to(torch_device).eval()
            with torch.no_grad():
                actual = reloaded(**inputs)[0]
            torch.testing.assert_close(actual, expected)

    def test_audio_input(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()

        input_ids = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size)
        input_ids[:, 1:5] = config.audio_token_id
        modality_ids = input_ids.new_ones(input_ids.shape)
        modality_ids[:, 1:5] = 2
        input_features = floats_tensor([self.model_tester.batch_size, 32, 8])
        feature_mask = input_ids.new_ones((self.model_tester.batch_size, 32))

        outputs = model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_attention_mask=feature_mask,
            modality_ids=modality_ids,
        )

        self.assertEqual(outputs.logits.shape, (self.model_tester.batch_size, self.model_tester.seq_length, 32))
        self.assertEqual(outputs.audio_hidden_states.shape, (self.model_tester.batch_size * 4, 32))
        self.assertEqual(model.model.conformer.config._attn_implementation, "eager")
        expected_inv_freq = torch.exp(
            torch.arange(0, config.encoder.hidden_size, 2, device=torch_device, dtype=torch.float32)
            * -(math.log(10_000.0) / config.encoder.hidden_size)
        )
        torch.testing.assert_close(
            model.model.conformer.encode_positions.inv_freq, expected_inv_freq, atol=0.0, rtol=0.0
        )

    def test_audio_only_targets_have_finite_loss(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).train()
        labels = torch.full_like(inputs["input_ids"], -100)
        audio_labels = labels.new_full((*labels.shape, config.codebooks), -100)
        audio_labels[:, 2:4] = 1
        outputs = model(**inputs, labels=labels, audio_labels=audio_labels, use_cache=False)
        torch.testing.assert_close(outputs.loss, outputs.audio_loss)
        self.assertEqual(outputs.text_loss.item(), 0.0)
        outputs.loss.backward()
        self.assertTrue(torch.isfinite(model.model.depth_linear.weight.grad).all())
        self.assertGreater(model.model.depth_linear.weight.grad.abs().sum().item(), 0)

    def test_all_ignored_targets_have_zero_loss(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).train()
        labels = torch.full_like(inputs["input_ids"], -100)
        audio_labels = labels.new_full((*labels.shape, config.codebooks), -100)
        outputs = model(**inputs, labels=labels, audio_labels=audio_labels, use_cache=False)
        self.assertEqual(outputs.loss.item(), 0.0)
        outputs.loss.backward()
        self.assertTrue(torch.isfinite(model.get_input_embeddings().weight.grad).all())

    def test_generation_config_length_and_override(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()
        inputs = torch.tensor([[1, 2]], device=torch_device)
        model.generation_config.max_new_tokens = 1
        with patch.object(model, "_sample", return_value=inputs.new_tensor([2])):
            self.assertEqual(model.generate(input_ids=inputs).modalities.shape[-1], 1)
            explicit = GenerationConfig(max_new_tokens=2)
            self.assertEqual(model.generate(input_ids=inputs, generation_config=explicit).modalities.shape[-1], 2)
            self.assertEqual(model.generate(input_ids=inputs, max_new_tokens=3).modalities.shape[-1], 3)
        self.assertEqual(model.generation_config.max_new_tokens, 1)

    def test_generation_config_sampling_and_eos(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()
        input_ids = torch.tensor([[1, 2]], device=torch_device)
        settings = GenerationConfig(max_new_tokens=3, do_sample=True, temperature=0.7, top_k=5, eos_token_id=2)
        with patch.object(model, "_sample", return_value=input_ids.new_tensor([2])) as sampler:
            result = model.generate(input_ids=input_ids, generation_config=settings)
        self.assertEqual(result.sequences.tolist(), [[2]])
        self.assertEqual(sampler.call_args.kwargs, {"temperature": 0.7, "top_k": 5})
        with self.assertRaisesRegex(ValueError, "top_p"):
            model.generate(input_ids=input_ids, do_sample=True, top_p=0.9)

    def test_compile_fullgraph_audio_input(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).eval()
        inputs = {
            "input_ids": torch.tensor([[1, 6, 2, 2]]),
            "input_features": torch.randn(1, 8, 8),
            "input_features_attention_mask": torch.ones(1, 8, dtype=torch.long),
            "use_cache": False,
        }
        with torch.no_grad(), torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            expected = model(**inputs).logits
            actual = torch.compile(model, fullgraph=True, backend="eager")(**inputs).logits
        torch.testing.assert_close(actual, expected)
        torch._dynamo.reset()

    def test_depth_attention_backends(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).eval()
        depth = model.model.depthformer
        hidden_states = torch.randn(1, config.codebooks, config.depthformer.dim)
        with torch.no_grad():
            expected, _ = depth(hidden_states)
            depth.config._attn_implementation = "eager"
            actual, _ = depth(hidden_states)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_multimodal_training_gradients(self):
        config = self.model_tester.get_config()
        for checkpointing in (False, True):
            model = Lfm2AudioForConditionalGeneration(config).to(torch_device).train()
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            input_ids = torch.tensor([[1, 6, 2, 2]], device=torch_device).repeat(2, 1)
            inputs = {
                "input_ids": input_ids,
                "modality_ids": torch.tensor([[1, 2, 3, 1]], device=torch_device).repeat(2, 1),
                "input_features": floats_tensor([2, 8, 8]),
                "input_features_attention_mask": torch.ones(2, 8, device=torch_device, dtype=torch.long),
                "audio_codes": torch.ones(2, 2, 1, device=torch_device, dtype=torch.long),
                "audio_labels": torch.ones(2, 4, 2, device=torch_device, dtype=torch.long),
                "labels": input_ids,
            }
            loss = model(**inputs, use_cache=False).loss
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_compile_fullgraph_text_and_audio_output(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).eval()
        input_ids = torch.tensor([[1, 2, 2]])
        for extra in (
            {},
            {"modality_ids": torch.tensor([[1, 3, 3]]), "audio_codes": torch.ones(1, 2, 2, dtype=torch.long)},
        ):
            with torch.no_grad():
                expected = model(input_ids=input_ids, use_cache=False, **extra).logits
                compiled = torch.compile(model, fullgraph=True, backend="eager")
                actual = compiled(input_ids=input_ids, use_cache=False, **extra).logits
            torch.testing.assert_close(actual, expected)
        torch._dynamo.reset()

    def test_audio_loss_is_shifted(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()
        input_ids = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size)
        input_ids[input_ids == config.audio_token_id] = 2
        audio_labels = input_ids.new_full((*input_ids.shape, config.codebooks), -100)
        audio_labels[:, 2:4] = ids_tensor([self.model_tester.batch_size, 2, config.codebooks], config.audio_vocab_size)

        outputs = model(input_ids=input_ids, audio_labels=audio_labels)

        self.assertEqual(outputs.audio_logits.shape, (self.model_tester.batch_size * 2, 2, 17))
        self.assertTrue(outputs.audio_loss.isfinite())

    def test_combined_loss_is_weighted_by_supervised_tokens(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()
        input_ids = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size)
        input_ids[input_ids == config.audio_token_id] = 2
        labels = input_ids.clone()
        labels[:, 0] = -100
        audio_labels = input_ids.new_full((*input_ids.shape, config.codebooks), -100)
        audio_labels[:, 2:4] = ids_tensor([self.model_tester.batch_size, 2, config.codebooks], config.audio_vocab_size)

        outputs = model(input_ids=input_ids, labels=labels, audio_labels=audio_labels)

        text_tokens = (labels[:, 1:] != -100).sum()
        audio_tokens = self.model_tester.batch_size * 2
        expected = (outputs.text_loss * text_tokens + outputs.audio_loss * audio_tokens) / (text_tokens + audio_tokens)
        torch.testing.assert_close(outputs.loss, expected)

    def test_depthformer_cache_matches_full_forward(self):
        config = self.model_tester.get_config()
        depthformer = Lfm2AudioForConditionalGeneration(config).model.depthformer.to(torch_device).eval()
        hidden_states = floats_tensor([self.model_tester.batch_size, config.codebooks, config.depthformer.dim])

        full_output, _ = depthformer(hidden_states)
        cached_output = []
        cache = None
        for position in range(config.codebooks):
            output, cache = depthformer(
                hidden_states[:, position : position + 1], past_key_values=cache, use_cache=True
            )
            cached_output.append(output)

        torch.testing.assert_close(full_output, torch.cat(cached_output, dim=1), atol=1e-5, rtol=1e-5)

    def test_depthformer_cached_chunk_matches_full_forward(self):
        config = self.model_tester.get_config()
        depthformer = Lfm2AudioForConditionalGeneration(config).model.depthformer.to(torch_device).eval()
        hidden_states = floats_tensor([2, 5, config.depthformer.dim])
        with torch.no_grad():
            expected, _ = depthformer(hidden_states)
            first, cache = depthformer(hidden_states[:, :2], use_cache=True)
            second, _ = depthformer(hidden_states[:, 2:], past_key_values=cache, use_cache=True)
        torch.testing.assert_close(torch.cat([first, second], dim=1), expected, atol=1e-5, rtol=1e-5)

    def test_sequential_generation_switches_to_audio(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()
        input_ids = ids_tensor([1, 3], config.vocab_size)
        sampled_tokens = [
            input_ids.new_tensor([config.audio_start_token_id]),
            input_ids.new_tensor([1]),
            input_ids.new_tensor([2]),
        ]

        with patch.object(model, "_sample", side_effect=sampled_tokens):
            output = model.generate(input_ids=input_ids, max_new_tokens=2)

        self.assertEqual(output.sequences.tolist(), [[config.audio_start_token_id]])
        self.assertEqual(output.audio_codes.shape, (1, config.codebooks, 1))
        self.assertEqual(output.modalities.tolist(), [[1, 3]])

    def test_interleaved_generation_does_not_return_terminal_eos(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()
        input_ids = ids_tensor([1, 3], config.vocab_size)
        input_ids[input_ids == config.audio_token_id] = 2

        with patch.object(model, "_sample", return_value=input_ids.new_tensor([config.eos_token_id])):
            output = model.generate(input_ids=input_ids, max_new_tokens=1, generation_mode="interleaved")

        self.assertEqual(output.sequences.shape, (1, 0))
        self.assertEqual(output.audio_codes.shape, (1, config.codebooks, 0))
        self.assertEqual(output.modalities.shape, (1, 0))


@require_torch
class Lfm2AudioDetokenizerTest(unittest.TestCase):
    all_model_classes = (Lfm2AudioDetokenizer,)

    def test_missing_window_is_initialized(self):
        from safetensors.torch import load_file, save_file

        config = Lfm2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            layer_types=["full_attention"],
            output_size=1282,
        )
        with tempfile.TemporaryDirectory() as folder:
            model = Lfm2AudioDetokenizer(config)
            expected = model.istft.window.clone()
            model.save_pretrained(folder)
            filename = Path(folder) / "model.safetensors"
            state_dict = load_file(filename)
            del state_dict["istft.window"]
            save_file(state_dict, filename, metadata={"format": "pt"})
            reloaded = Lfm2AudioDetokenizer.from_pretrained(folder)
        torch.testing.assert_close(reloaded.istft.window, expected, atol=0, rtol=0)

    def test_forward(self):
        config = Lfm2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            layer_types=["full_attention"],
            max_position_embeddings=64,
            output_size=1282,
        )
        audio_codes = ids_tensor([1, 8, 2], 2048).to(torch_device)

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            waveform = model(audio_codes)

            self.assertEqual(model.lfm.config._attn_implementation, "sdpa")
            self.assertEqual(waveform.shape, (1, 3840))
            self.assertTrue(waveform.isfinite().all())


@slow
@require_torch_gpu
@require_librosa
@require_torchaudio
class Lfm2AudioIntegrationTest(unittest.TestCase):
    """Expectations come from reproduce_integration_outputs.py and the unmodified Liquid Audio checkpoint.

    To test a local conversion without uploading it::

        LFM2_AUDIO_CHECKPOINT=/path/to/LFM2.5-Audio-1.5B-hf RUN_SLOW=1 pytest \
            tests/models/lfm2_audio/test_modeling_lfm2_audio.py::Lfm2AudioIntegrationTest
    """

    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads((Path(__file__).parent / "fixtures" / "integration_expected.json").read_text())
        checkpoint = os.environ.get("LFM2_AUDIO_CHECKPOINT", "kadirnar/LFM2.5-Audio-1.5B-hf")
        revision = os.environ.get(
            "LFM2_AUDIO_CHECKPOINT_REVISION",
            "24bb1668e710037bb64209c35d71a6376d2d7d13" if checkpoint == "kadirnar/LFM2.5-Audio-1.5B-hf" else None,
        )
        cls.processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)
        cls.model, loading_info = AutoModelForMultimodalLM.from_pretrained(
            checkpoint, revision=revision, dtype=torch.bfloat16, output_loading_info=True
        )
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys"):
            if loading_info[name]:
                raise AssertionError(f"Converted checkpoint has {name}: {loading_info[name]}")
        cls.model = cls.model.to(torch_device).eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model, cls.processor
        cleanup(torch_device, gc_collect=True)

    @parameterized.expand(["asr", "tts_us_male", "tts_us_female", "tts_uk_male", "tts_uk_female", "interleaved"])
    def test_reference_generation(self, name):
        import soundfile as sf
        import torchaudio

        case = next(case for case in self.expected["cases"] if case["name"] == name)
        if "audio_url" in case:
            with urlopen(case["audio_url"]) as response:
                audio_bytes = response.read()
            self.assertEqual(hashlib.sha256(audio_bytes).hexdigest(), case["audio_sha256"])
            audio, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            # Match the reference's GPU torchaudio resampling before running the native frontend.
            audio = (
                torchaudio.functional.resample(
                    torch.from_numpy(audio).to(torch_device),
                    sampling_rate,
                    self.processor.feature_extractor.sampling_rate,
                )
                .cpu()
                .numpy()
            )
            inputs = self.processor.apply_transcription_request(audio, prompt=case["prompt"], device=torch_device)
        else:
            inputs = self.processor.apply_text_to_speech_request(case["text"], prompt=case["prompt"])
        inputs = inputs.to(device=torch_device, dtype=self.model.dtype)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_mode=case["mode"],
                max_new_tokens=case["max_new_tokens"],
                text_top_k=1,
                audio_top_k=1,
            )
        self.assertEqual(outputs.sequences[0].tolist(), case["sequences"])
        self.assertEqual(outputs.modalities[0].tolist(), case["modalities"])
        self.assertEqual(outputs.audio_codes[0].T.tolist(), case["audio_codes"])
        self.assertEqual(
            self.processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), case["decoded_text"]
        )
        if "waveform_shape" in case:
            waveform = self.processor.decode_audio(outputs.audio_codes).float().cpu()
            self.assertEqual(list(waveform.shape), case["waveform_shape"])
            torch.testing.assert_close(
                waveform[0, 1000:1032], torch.tensor(case["waveform_slice"]), atol=1e-4, rtol=1e-4
            )
            self.assertAlmostEqual(waveform.mean().item(), case["waveform_mean"], delta=1e-4)
            self.assertAlmostEqual(waveform.std().item(), case["waveform_std"], delta=1e-4)


if __name__ == "__main__":
    unittest.main()
