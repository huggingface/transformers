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
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import torch
from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import (
    AutoModel,
    Qwen3TTSTokenizerConfig,
    Qwen3TTSTokenizerModel,
    Qwen3TTSTokenizerQuantizerConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin


class Qwen3TTSTokenizerModelTester:
    """
    Builds a tiny Qwen3TTSTokenizer config and synthetic inputs for unit testing.
    """

    def __init__(
        self,
        parent,
        batch_size=2,
        num_quantizers=4,
        audio_samples=960,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_quantizers = num_quantizers
        self.audio_samples = audio_samples
        self.is_training = is_training

        self.config_kwargs = {
            "encoder_hidden_size": 16,
            "encoder_num_hidden_layers": 1,
            "encoder_num_attention_heads": 2,
            "encoder_num_key_value_heads": 2,
            "encoder_intermediate_size": 32,
            "encoder_num_filters": 8,
            "encoder_kernel_size": 7,
            "encoder_residual_kernel_size": 3,
            "encoder_last_kernel_size": 3,
            "encoder_num_residual_layers": 1,
            "encoder_upsampling_ratios": [8, 6],
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 32,
            "codebook_dim": 16,
            "latent_dim": 16,
            "decoder_dim": 32,
            "upsample_rates": [2, 12],
            "upsampling_ratios": [2, 12],
            "quantizer_config": {
                "codebook_size": 8,
                "codebook_dim": 8,
                "vector_quantization_hidden_dimension": 8,
                "hidden_size": 16,
                "n_codebooks": num_quantizers,
                "semantic_quantizer_config": {"n_codebooks": 1},
            },
        }

    def get_config(self, **kwargs):
        return Qwen3TTSTokenizerConfig(**{**copy.deepcopy(self.config_kwargs), **kwargs})

    def prepare_config_and_inputs(self):
        input_values = torch.randn([self.batch_size, 1, self.audio_samples], device=torch_device)
        padding_mask = torch.ones([self.batch_size, self.audio_samples], dtype=torch.bool, device=torch_device)
        config = self.get_config()
        return config, {"input_values": input_values, "padding_mask": padding_mask}

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()

    def create_and_check_model_forward(self, config, inputs_dict):
        model = Qwen3TTSTokenizerModel(config=config).to(torch_device).eval()
        result = model(**inputs_dict)
        self.parent.assertEqual(result.audio_values.shape, inputs_dict["padding_mask"].shape)


if is_torch_available():
    import torch

    from transformers.models.qwen3_tts_tokenizer.modeling_qwen3_tts_tokenizer import (
        Qwen3TTSTokenizerSplitResidualVectorQuantizer,
    )


@require_torch
class Qwen3TTSTokenizerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Qwen3TTSTokenizerModel,) if is_torch_available() else ()
    _is_composite = True
    is_encoder_decoder = True
    test_pruning = False
    test_resize_embeddings = False
    test_torch_exportable = False  # data-dependent guard in codec attention padding
    test_head_masking = False
    test_missing_keys = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        inputs_dict.pop("output_attentions", None)
        inputs_dict.pop("output_hidden_states", None)
        return inputs_dict

    def setUp(self):
        self.model_tester = Qwen3TTSTokenizerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3TTSTokenizerConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_quantizer_config_rejects_invalid_codebook_counts(self):
        for total_count, semantic_count in ((-1, 1), (0, 1), (3, -1), (3, 0), (3, 4)):
            with (
                self.subTest(total_count=total_count, semantic_count=semantic_count),
                self.assertRaises(StrictDataclassClassValidationError),
            ):
                Qwen3TTSTokenizerQuantizerConfig(
                    n_codebooks=total_count, semantic_quantizer_config={"n_codebooks": semantic_count}
                )

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_component_save_load_preserves_outputs(self):
        _, inputs = self.model_tester.prepare_config_and_inputs()
        config = self.model_tester.get_config(
            encoder_num_hidden_layers=2,
            encoder_num_attention_heads=4,
            encoder_num_key_value_heads=2,
            encoder_head_dim=4,
            encoder_intermediate_size=24,
            encoder_upsampling_ratios=(8, 6),
            hidden_size=24,
            num_attention_heads=3,
            num_key_value_heads=1,
        )
        model = Qwen3TTSTokenizerModel(config).to(torch_device).eval()
        with torch.no_grad():
            for module in model.quantizer.modules():
                if hasattr(module, "embed_sum"):
                    module.embed_sum.normal_()
            expected_outputs = model(**inputs)
            embeddings = model.encoder(inputs["input_values"]).last_hidden_state
            codes = model.quantizer.encode(embeddings)
            quantized = model.quantizer.decode(codes.transpose(0, 1))
        components = ((model.encoder, inputs["input_values"]), (model.decoder, quantized))

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            projected_configs = [type(component.config).from_pretrained(directory) for component, _ in components]
            restored, loading_info = AutoModel.from_pretrained(directory, output_loading_info=True)
            restored = restored.to(torch_device).eval()
            self.assertFalse(loading_info["missing_keys"])
            self.assertFalse(loading_info["unexpected_keys"])
            self.assertFalse(loading_info["mismatched_keys"])
            with torch.no_grad():
                actual_outputs = restored(**inputs)
            torch.testing.assert_close(actual_outputs.audio_codes, expected_outputs.audio_codes)
            torch.testing.assert_close(actual_outputs.audio_values, expected_outputs.audio_values)
            torch.testing.assert_close(actual_outputs.audio_codes_mask, expected_outputs.audio_codes_mask)

        for (component, component_input), projected_config in zip(components, projected_configs):
            with self.subTest(component=type(component).__name__), tempfile.TemporaryDirectory() as directory:
                component.save_pretrained(directory)
                restored, loading_info = AutoModel.from_pretrained(directory, output_loading_info=True)
                restored = restored.to(torch_device).eval()
                self.assertFalse(loading_info["missing_keys"])
                self.assertFalse(loading_info["unexpected_keys"])
                self.assertFalse(loading_info["mismatched_keys"])
                projected = AutoModel.from_config(projected_config).to(torch_device).eval()
                projected.load_state_dict(component.state_dict())
                with torch.no_grad():
                    expected = component(component_input)
                    actual = restored(component_input)
                    projected_output = projected(component_input)
                if component is model.encoder:
                    expected = expected.last_hidden_state
                    actual = actual.last_hidden_state
                    projected_output = projected_output.last_hidden_state
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(projected_output, expected)

    def test_published_nested_config_preserves_outputs_and_canonical_overrides(self):
        # A tiny version of the published checkpoint schema, including unused Mimi quantizer settings.
        published_config = {
            "model_type": "qwen3_tts_tokenizer_12hz",
            "encoder_valid_num_quantizers": 4,
            "input_sample_rate": 24000,
            "output_sample_rate": 24000,
            "encoder_config": {
                "_frame_rate": 125,
                "sampling_rate": 24000,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "intermediate_size": 24,
                "num_filters": 8,
                "kernel_size": 7,
                "residual_kernel_size": 3,
                "last_kernel_size": 3,
                "num_residual_layers": 1,
                "upsampling_ratios": [8, 6],
                "rope_theta": 2.0,
                "layer_scale_initial_scale": 1.0,
                "codebook_size": 13,
                "codebook_dim": 4,
                "vector_quantization_hidden_dimension": 4,
                "num_quantizers": 8,
                "num_semantic_quantizers": 1,
                "upsample_groups": 8,
            },
            "decoder_config": {
                "hidden_size": 24,
                "num_hidden_layers": 1,
                "num_attention_heads": 3,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "intermediate_size": 32,
                "codebook_dim": 16,
                "latent_dim": 16,
                "decoder_dim": 32,
                "upsample_rates": [2, 12],
                "upsampling_ratios": [2, 12],
                "rope_theta": 4.0,
                "layer_scale_initial_scale": 1.0,
                "codebook_size": 8,
                "num_quantizers": 4,
                "num_semantic_quantizers": 1,
                "semantic_codebook_size": 11,
                "vector_quantization_hidden_dimension": 16,
            },
        }
        canonical_settings = {
            "encoder_num_hidden_layers": 2,
            "encoder_num_attention_heads": 4,
            "encoder_num_key_value_heads": 2,
            "encoder_head_dim": 4,
            "encoder_intermediate_size": 24,
            "encoder_frame_rate": 125,
            "encoder_rope_parameters": {"rope_type": "default", "rope_theta": 2.0},
            "encoder_layer_scale_initial_scale": 1.0,
            "rope_parameters": {"rope_type": "default", "rope_theta": 4.0},
            "layer_scale_initial_scale": 1.0,
            "hidden_size": 24,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
        }
        overrides = {
            "encoder_intermediate_size": 40,
            "intermediate_size": 48,
            "quantizer_config": {
                "codebook_size": 7,
                "codebook_dim": 4,
                "vector_quantization_hidden_dimension": 4,
                "hidden_size": 16,
                "n_codebooks": 3,
                "semantic_quantizer_config": {"n_codebooks": 2, "codebook_size": 5},
            },
        }
        input_values = torch.linspace(
            -1, 1, self.model_tester.batch_size * self.model_tester.audio_samples, device=torch_device
        ).reshape(self.model_tester.batch_size, 1, self.model_tester.audio_samples)
        inputs = {
            "input_values": input_values,
            "padding_mask": torch.ones_like(input_values[:, 0], dtype=torch.bool),
        }
        for override_source in (None, "dictionary", "kwargs"):
            with self.subTest(override_source=override_source), tempfile.TemporaryDirectory() as directory:
                config_dict = copy.deepcopy(published_config)
                explicit_settings = overrides if override_source is not None else {}
                kwargs = {}
                if override_source == "dictionary":
                    config_dict.update(copy.deepcopy(overrides))
                elif override_source == "kwargs":
                    config_dict.update({"encoder_intermediate_size": 36, "intermediate_size": 44})
                    kwargs = copy.deepcopy(overrides)
                original_dict = copy.deepcopy(config_dict)
                original_kwargs = copy.deepcopy(kwargs)
                normalized_config = Qwen3TTSTokenizerConfig.from_dict(config_dict, **kwargs)
                self.assertEqual(config_dict, original_dict)
                self.assertEqual(kwargs, original_kwargs)

                set_seed(0)
                reference_config = self.model_tester.get_config(**{**canonical_settings, **explicit_settings})
                reference = Qwen3TTSTokenizerModel(reference_config).to(torch_device).eval()
                with torch.no_grad():
                    for module in reference.quantizer.modules():
                        if hasattr(module, "embed_sum"):
                            module.embed_sum.normal_()
                    expected = reference(**inputs)
                    expected_embeddings = reference.encoder(input_values).last_hidden_state
                model = Qwen3TTSTokenizerModel(normalized_config).to(torch_device).eval()
                model.load_state_dict(reference.state_dict())
                with torch.no_grad():
                    actual = model(**inputs)
                    actual_embeddings = model.encoder(input_values).last_hidden_state
                torch.testing.assert_close(actual_embeddings, expected_embeddings, atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(actual.audio_codes, expected.audio_codes)
                torch.testing.assert_close(actual.audio_values, expected.audio_values, atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(actual.audio_codes_mask, expected.audio_codes_mask)

                model.save_pretrained(directory)
                restored, loading_info = AutoModel.from_pretrained(directory, output_loading_info=True)
                restored = restored.to(torch_device).eval()
                self.assertFalse(loading_info["missing_keys"])
                self.assertFalse(loading_info["unexpected_keys"])
                self.assertFalse(loading_info["mismatched_keys"])
                with torch.no_grad():
                    actual = restored(**inputs)
                    actual_embeddings = restored.encoder(input_values).last_hidden_state
                torch.testing.assert_close(actual_embeddings, expected_embeddings, atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(actual.audio_codes, expected.audio_codes)
                torch.testing.assert_close(actual.audio_values, expected.audio_values, atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(actual.audio_codes_mask, expected.audio_codes_mask)
                self.assertEqual(config_dict, original_dict)
                self.assertEqual(kwargs, original_kwargs)

    def test_independent_semantic_quantizer_encode_decode(self):
        config = Qwen3TTSTokenizerQuantizerConfig(
            n_codebooks=3,
            codebook_size=5,
            codebook_dim=2,
            vector_quantization_hidden_dimension=2,
            hidden_size=2,
            semantic_quantizer_config={
                "n_codebooks": 2,
                "codebook_size": 3,
                "codebook_dim": 1,
                "vector_quantization_hidden_dimension": 1,
            },
        )
        quantizer = Qwen3TTSTokenizerSplitResidualVectorQuantizer(config).eval()
        semantic = quantizer.semantic_residual_vector_quantizer
        with torch.no_grad():
            semantic.input_proj.weight.copy_(torch.tensor([[[1.0], [0.0]]]))
            semantic.output_proj.weight.copy_(torch.tensor([[[1.0]], [[2.0]]]))
            for layer in semantic.layers:
                layer.codebook.embed_sum.copy_(torch.tensor([[0.0], [2.0], [4.0]]))
            quantizer.acoustic_residual_vector_quantizer.layers[0].codebook.embed_sum.copy_(
                torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 1.0], [5.0, 3.0], [7.0, 5.0]])
            )
            embeddings = torch.tensor([[[3.25, 7.25], [1.0, 5.0]]])
            codes = quantizer.encode(embeddings)
            expected_codes = torch.tensor([[[2, 2]], [[0, 2]], [[2, 4]]])
            torch.testing.assert_close(codes, expected_codes)
            torch.testing.assert_close(
                quantizer.decode(codes.transpose(0, 1)), torch.tensor([[[7.0, 15.0], [9.0, 21.0]]])
            )
            semantic_codes = quantizer.encode(embeddings, num_quantizers=2)
            torch.testing.assert_close(semantic_codes, expected_codes[:2])
            torch.testing.assert_close(
                quantizer.decode(semantic_codes.transpose(0, 1)), torch.tensor([[[4.0, 8.0], [8.0, 16.0]]])
            )
        for invalid_count in (1, 4):
            with self.subTest(num_quantizers=invalid_count), self.assertRaises(ValueError):
                quantizer.encode(embeddings, num_quantizers=invalid_count)

    def test_semantic_quantizer_save_load_preserves_outputs(self):
        for legacy_config in (False, True):
            with self.subTest(legacy_config=legacy_config), tempfile.TemporaryDirectory() as directory:
                config, inputs = self.model_tester.prepare_config_and_inputs()
                config_dict = config.to_dict()
                quantizer_config = config_dict["quantizer_config"]
                if legacy_config:
                    quantizer_config.pop("semantic_quantizer_config")
                    quantizer_config["num_semantic_quantizers"] = 2
                else:
                    quantizer_config["semantic_quantizer_config"] = {
                        "n_codebooks": 2,
                        "codebook_size": 5,
                        "codebook_dim": 4,
                        "vector_quantization_hidden_dimension": 4,
                    }
                model = Qwen3TTSTokenizerModel(Qwen3TTSTokenizerConfig.from_dict(config_dict))
                model = model.to(torch_device).eval()
                with torch.no_grad():
                    for module in model.quantizer.modules():
                        if hasattr(module, "embed_sum"):
                            module.embed_sum.normal_()
                    expected = model(**inputs)
                model.save_pretrained(directory)
                with open(Path(directory) / "config.json", encoding="utf-8") as config_file:
                    saved_quantizer_config = json.load(config_file)["quantizer_config"]
                self.assertNotIn("num_semantic_quantizers", saved_quantizer_config)
                self.assertEqual(saved_quantizer_config["semantic_quantizer_config"]["n_codebooks"], 2)
                restored, loading_info = AutoModel.from_pretrained(directory, output_loading_info=True)
                self.assertFalse(loading_info["missing_keys"])
                self.assertFalse(loading_info["unexpected_keys"])
                self.assertFalse(loading_info["mismatched_keys"])
                restored = restored.to(torch_device).eval()
                with torch.no_grad():
                    actual = restored(**inputs)
                torch.testing.assert_close(actual.audio_codes, expected.audio_codes)
                torch.testing.assert_close(actual.audio_values, expected.audio_values)

    def test_forward_returns_encode_codes_and_truncates_audio(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        inputs_dict["padding_mask"][0, self.model_tester.audio_samples // 2 :] = False
        model = Qwen3TTSTokenizerModel(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(**inputs_dict)
            encoded = model.encode(**inputs_dict)

        torch.testing.assert_close(outputs.audio_codes, encoded.audio_codes)
        torch.testing.assert_close(outputs.audio_codes_mask, encoded.audio_codes_mask)
        self.assertLess(outputs.audio_codes_mask[0].sum(), outputs.audio_codes_mask[1].sum())
        self.assertEqual(outputs.audio_values.shape, inputs_dict["padding_mask"].shape)

    def test_decode_latents_matches_codes_and_preserves_code_precedence(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = Qwen3TTSTokenizerModel(config).to(torch_device).eval()
        with torch.no_grad():
            for module in model.quantizer.modules():
                if hasattr(module, "embed_sum"):
                    module.embed_sum.normal_()
            codes = model.encode(**inputs).audio_codes
            latents = model.quantizer.decode(codes.transpose(1, 2))
            expected = model.decode(codes).audio_values
            actual = model.decode(latents=latents).audio_values
            tuple_output = model.decode(latents=latents, return_dict=False)
            codes_with_latents = model.decode(codes, latents=torch.full_like(latents, float("nan"))).audio_values

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(tuple_output[0], expected)
        torch.testing.assert_close(codes_with_latents, expected)

    def test_decode_requires_codes_or_latents(self):
        model = Qwen3TTSTokenizerModel(self.model_tester.get_config()).to(torch_device).eval()
        with self.assertRaises(ValueError):
            model.decode()

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                tuple_outputs = model(**inputs_dict, return_dict=False)
                dict_outputs = model(**inputs_dict, return_dict=True).to_tuple()

            self.assertEqual(len(tuple_outputs), len(dict_outputs))
            for tuple_output, dict_output in zip(tuple_outputs, dict_outputs):
                torch.testing.assert_close(tuple_output, dict_output)

    def test_batching_equivalence(self):
        super().test_batching_equivalence(atol=5e-4, rtol=1e-3)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            signature = inspect.signature(model_class(config).forward)
            arg_names = [*signature.parameters.keys()]
            self.assertListEqual(arg_names[:2], ["input_values", "padding_mask"])

    @unittest.skip(
        reason="`_init_weights` does not cover the EuclideanCodebook buffers (`embed_sum`, `cluster_usage`), "
        "so they cannot be reinitialized from the meta device"
    )
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip(reason="The codec forward method does not expose submodel capture outputs.")
    def test_capture_outputs_decorator(self):
        pass

    @unittest.skip(reason="The codec forward method does not expose submodel hidden states.")
    def test_can_capture_specific_layers_hidden_states(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerModel does not have `inputs_embeds` logic")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="codec model has no token embeddings, so `get_input_embeddings` is not implemented")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerModel does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerModel does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerModel does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass


@require_torch
class Qwen3TTSTokenizerIntegrationTest(unittest.TestCase):
    """
    Slow integration tests against the original checkpoint.
    """

    TARGET_SAMPLE_RATE = 24000

    @classmethod
    def setUpClass(cls):
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

    def tearDown(self):
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)

    def _load_fixture(self, name):
        path = Path(__file__).parent.parent.parent / f"fixtures/qwen3_tts_tokenizer/{name}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_datasamples(self, num_samples):
        """Decode with soundfile and resample with torchaudio.

        The reproducer loads the same samples the same way, so both sides see bit-identical
        audio: resampling differences between backends would change the codes.
        """
        import io

        import numpy as np
        import soundfile as sf
        import torchaudio
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        samples = []
        for raw in ds.data.column("audio").to_pylist()[:num_samples]:
            audio_bytes = raw.get("bytes")
            array, sr = (
                sf.read(io.BytesIO(audio_bytes), dtype="float32")
                if audio_bytes
                else sf.read(raw["path"], dtype="float32")
            )
            if array.ndim > 1:
                array = array.mean(axis=1)
            if sr != self.TARGET_SAMPLE_RATE:
                array = torchaudio.functional.resample(torch.from_numpy(array), sr, self.TARGET_SAMPLE_RATE).numpy()
            samples.append(np.array(array, dtype=np.float32))
        return samples

    @slow
    def test_single(self):
        """
        Ground truth generated from the original Qwen3-TTS tokenizer.

        reproducer: https://gist.github.com/ShahVandit/cab13f3b7232c52b4ff93cce592950c4#file-reproducer_qwen3_tts_multicodebook_tokenizer-py
        """
        set_seed(42)
        expected = self._load_fixture("expected_results_single.json")

        model = Qwen3TTSTokenizerModel.from_pretrained(
            self.checkpoint, dtype=torch.float32, device_map=torch_device
        ).eval()

        samples = self._load_datasamples(1)
        input_values = torch.tensor(samples[0])[None, None].to(model.device)
        padding_mask = torch.ones(1, input_values.shape[-1], dtype=torch.bool, device=model.device)

        with torch.no_grad():
            encoded = model.encode(input_values, padding_mask=padding_mask)
            codes = encoded.audio_codes[0]  # [time, num_quantizers]

        torch.testing.assert_close(
            codes.cpu().long(),
            torch.tensor(expected["audio_codes"][0], dtype=torch.long),
        )

        with torch.no_grad():
            decoded = model.decode(codes.unsqueeze(0))

        expected_slice = torch.tensor(expected["audio_values_slice"][0], dtype=torch.float32)
        torch.testing.assert_close(
            decoded.audio_values[0].cpu().float()[..., : expected_slice.shape[-1]],
            expected_slice,
            atol=2e-2,
            rtol=1e-3,
        )

    @slow
    def test_batch(self):
        """
        Ground truth generated from the original Qwen3-TTS tokenizer.

        reproducer: https://gist.github.com/ShahVandit/cab13f3b7232c52b4ff93cce592950c4#file-reproducer_qwen3_tts_multicodebook_tokenizer-py
        """
        import numpy as np

        set_seed(42)
        expected = self._load_fixture("expected_results_batch.json")

        model = Qwen3TTSTokenizerModel.from_pretrained(
            self.checkpoint, dtype=torch.float32, device_map=torch_device
        ).eval()

        samples = self._load_datasamples(2)
        max_len = max(len(sample) for sample in samples)
        input_values = (
            torch.stack([torch.tensor(np.pad(sample, (0, max_len - len(sample)))) for sample in samples])
            .unsqueeze(1)
            .to(model.device)
        )
        padding_mask = torch.stack(
            [
                torch.tensor(np.concatenate([np.ones(len(sample)), np.zeros(max_len - len(sample))]).astype(bool))
                for sample in samples
            ]
        ).to(model.device)

        with torch.no_grad():
            encoded = model.encode(input_values, padding_mask=padding_mask)
        codes_list = [codes[mask.bool()] for codes, mask in zip(encoded.audio_codes, encoded.audio_codes_mask)]

        for i, exp_codes in enumerate(expected["audio_codes"]):
            torch.testing.assert_close(
                codes_list[i].cpu().long(),
                torch.tensor(exp_codes, dtype=torch.long),
            )

        for i, (codes, exp_slice) in enumerate(zip(codes_list, expected["audio_values_slice"])):
            with torch.no_grad():
                decoded = model.decode(codes.unsqueeze(0))
            expected_slice = torch.tensor(exp_slice, dtype=torch.float32)
            torch.testing.assert_close(
                decoded.audio_values[0].cpu().float()[..., : expected_slice.shape[-1]],
                expected_slice,
                atol=2e-2,
                rtol=1e-3,
            )
