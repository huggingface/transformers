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

import unittest

from parameterized import parameterized

from transformers import (
    AutoProcessor,
    MossTranscribeDiarizeConfig,
    MossTranscribeDiarizeForConditionalGeneration,
    MossTranscribeDiarizeModel,
    Qwen3Config,
    WhisperConfig,
    is_torch_available,
)
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightRenaming, rename_source_key
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...alm_tester import ALMModelTest, ALMModelTester


if is_torch_available():
    import torch


class MossTranscribeDiarizeModelTester(ALMModelTester):
    config_class = MossTranscribeDiarizeConfig
    base_model_class = MossTranscribeDiarizeModel
    conditional_generation_class = MossTranscribeDiarizeForConditionalGeneration
    text_config_class = Qwen3Config
    audio_config_class = WhisperConfig
    audio_mask_key = None

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("feat_seq_length", 3000)
        kwargs.setdefault("d_model", 16)
        kwargs.setdefault("hidden_size", 16)
        kwargs.setdefault("intermediate_size", 32)
        kwargs.setdefault("encoder_layers", 1)
        kwargs.setdefault("encoder_attention_heads", 2)
        kwargs.setdefault("encoder_ffn_dim", 32)
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("max_position_embeddings", 64)
        kwargs.setdefault("audio_merge_size", 4)
        kwargs.setdefault("audio_token_id", 0)
        super().__init__(parent, **kwargs)

    def _prepare_modality_inputs(self, input_ids, config):
        num_audio_tokens = torch.full((self.batch_size,), 4, dtype=torch.long, device=torch_device)
        input_ids = self.place_audio_tokens(input_ids, config, num_audio_tokens)
        modality_inputs = {
            "input_features": self.create_audio_features(),
            "audio_feature_lengths": num_audio_tokens,
            "audio_chunk_mapping": torch.arange(self.batch_size, device=torch_device),
        }
        return input_ids, modality_inputs


@require_torch
class MossTranscribeDiarizeForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    model_tester_class = MossTranscribeDiarizeModelTester
    skip_test_audio_features_output_shape = True
    pipeline_model_mapping = (
        {"audio-text-to-text": MossTranscribeDiarizeForConditionalGeneration} if is_torch_available() else {}
    )

    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        return self.model_tester.encoder_layers

    def _audio_features_get_expected_num_hidden_states(self, model_tester=None):
        return self.model_tester.encoder_layers + 1

    @unittest.skip(
        reason="MossTranscribeDiarize replaces audio-token embeddings when input_features are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="MossTranscribeDiarize uses audio_feature_lengths/audio_chunk_mapping instead of audio masks.")
    def test_mismatching_num_audio_tokens(self):
        pass

    @unittest.skip(reason="Composite MOSS checkpoint loading is incompatible with meta-tensor cpu offload in CI.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Composite MOSS checkpoint loading is incompatible with disk offload in CI.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Composite MOSS checkpoint loading is incompatible with disk offload in CI.")
    def test_disk_offload_safetensors(self):
        pass

    def test_get_audio_features_groups_by_chunk_mapping(self):
        from unittest.mock import patch

        from transformers.modeling_outputs import BaseModelOutputWithPooling

        config = self.model_tester.config_class(
            text_config={
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 32,
            },
            audio_config={
                "num_mel_bins": 80,
                "d_model": 16,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 32,
                "max_source_positions": 16,
            },
            audio_merge_size=4,
            audio_token_id=0,
        )
        model = self.model_tester.base_model_class(config).to(torch_device)
        model.eval()

        merge_size = config.audio_merge_size
        hidden_size = config.audio_config.d_model
        whisper_seq_len = merge_size * 2
        num_chunks = 3
        input_features = torch.randn(
            num_chunks,
            config.audio_config.num_mel_bins,
            whisper_seq_len,
            device=torch_device,
            dtype=model.dtype,
        )
        audio_feature_lengths = torch.full((num_chunks,), 2, dtype=torch.long, device=torch_device)
        whisper_features = torch.stack(
            [
                torch.full((whisper_seq_len, hidden_size), float(chunk_idx + 1), device=torch_device, dtype=model.dtype)
                for chunk_idx in range(num_chunks)
            ]
        )

        projector_fingerprints = []

        def projector_forward(packed_features):
            fingerprint = packed_features[0, :, 0].sum().item()
            projector_fingerprints.append(fingerprint)
            return torch.full(
                (packed_features.shape[1], config.text_config.hidden_size),
                fingerprint,
                device=packed_features.device,
                dtype=packed_features.dtype,
            )

        contiguous_mapping = torch.tensor([0, 0, 1], dtype=torch.long, device=torch_device)
        shuffled_mapping = torch.tensor([1, 0, 0], dtype=torch.long, device=torch_device)

        with (
            patch.object(
                model.audio_tower,
                "forward",
                return_value=BaseModelOutputWithPooling(last_hidden_state=whisper_features),
            ),
            patch.object(model.multi_modal_projector, "forward", side_effect=projector_forward),
            torch.no_grad(),
        ):
            model.get_audio_features(
                input_features=input_features,
                audio_feature_lengths=audio_feature_lengths,
                audio_chunk_mapping=contiguous_mapping,
            )
            contiguous_fingerprints = list(projector_fingerprints)
            projector_fingerprints.clear()
            model.get_audio_features(
                input_features=input_features,
                audio_feature_lengths=audio_feature_lengths,
                audio_chunk_mapping=shuffled_mapping,
            )
            shuffled_fingerprints = list(projector_fingerprints)

        self.assertEqual(len(contiguous_fingerprints), 2)
        self.assertEqual(len(shuffled_fingerprints), 2)
        self.assertNotEqual(contiguous_fingerprints[0], shuffled_fingerprints[0])
        self.assertNotEqual(contiguous_fingerprints[1], shuffled_fingerprints[1])
        # Shuffled mapping groups chunks 1+2 into sample 0 and chunk 0 into sample 1.
        self.assertGreater(shuffled_fingerprints[0], shuffled_fingerprints[1])

    @unittest.skip(reason="MOSS generation compile coverage is not supported yet.")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip(reason="MOSS model-parallel generation is not supported in CI.")
    def test_model_parallel_beam_search(self):
        pass

    @unittest.skip(reason="MOSS model parallelism is not supported in CI.")
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="MOSS multi-GPU data parallel coverage is not supported in CI.")
    def test_multi_gpu_data_parallel_forward(self):
        pass


@require_torch
class MossTranscribeDiarizeModelConversionTest(unittest.TestCase):
    def test_skip_keys_device_placement_is_list(self):
        cfg = MossTranscribeDiarizeConfig(
            text_config={
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 32,
            },
            audio_config={
                "num_mel_bins": 80,
                "d_model": 16,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 32,
                "max_source_positions": 16,
            },
            audio_merge_size=4,
            audio_token_id=0,
        )
        model = MossTranscribeDiarizeForConditionalGeneration(cfg)
        self.assertIsInstance(model._skip_keys_device_placement, (list, set, tuple))
        self.assertIn("past_key_values", model._skip_keys_device_placement)
        self.assertEqual([x for x in model._skip_keys_device_placement if len(x) == 1], [])

    def test_original_checkpoint_keys_are_renamed(self):
        with torch.device("meta"):
            model = MossTranscribeDiarizeForConditionalGeneration(MossTranscribeDiarizeConfig())
        meta_state_dict = model.state_dict()
        renamings = [
            conversion
            for conversion in get_model_conversion_mapping(model)
            if isinstance(conversion, WeightRenaming)
        ]

        renamed_key, _ = rename_source_key(
            "model.whisper_encoder.conv1.weight",
            renamings,
            [],
            model.base_model_prefix,
            meta_state_dict,
        )
        self.assertEqual(renamed_key, "model.audio_tower.conv1.weight")

        renamed_key, _ = rename_source_key(
            "model.vq_adaptor.layers.0.weight",
            renamings,
            [],
            model.base_model_prefix,
            meta_state_dict,
        )
        self.assertEqual(renamed_key, "model.multi_modal_projector.linear_1.weight")


@require_torch
class MossTranscribeDiarizeForConditionalGenerationIntegrationTest(unittest.TestCase):
    checkpoint_name = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    audio_url = (
        "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
    )

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_transcription_request_generates_diarized_output(self):
        model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, dtype="auto"
        )
        inputs = self.processor.apply_transcription_request(self.audio_url).to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
        decoded = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]

        EXPECTED_TEXT = (
            "Mister quilter is apostle of the middle classes, and we are glad to welcome his gospel."
        )
        self.assertIn("[S01]", decoded)
        self.assertIn(EXPECTED_TEXT, decoded)

    @slow
    def test_audio_features_logits_are_finite(self):
        model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
            self.checkpoint_name, device_map=torch_device, dtype="auto"
        )
        inputs = self.processor.apply_transcription_request(self.audio_url).to(model.device, dtype=model.dtype)

        with torch.inference_mode():
            outputs = model(**inputs)

        self.assertTrue(torch.isfinite(outputs.logits).all())
        self.assertGreater(outputs.logits.shape[-1], 0)
