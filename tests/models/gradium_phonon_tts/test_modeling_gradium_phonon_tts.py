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
"""Testing suite for the PyTorch Gradium Phonon TTS model."""

import json
import os
import tempfile
import unittest

from transformers import GradiumPhononTTSConfig, GradiumPhononTTSFlowHeadConfig
from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ..pocket_tts import test_modeling_pocket_tts as pocket_tts


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForTextToWaveform,
        AutoProcessor,
        GradiumPhononTTSForConditionalGeneration,
        GradiumPhononTTSModel,
    )


class GradiumPhononTTSModelTester(pocket_tts.PocketTTSModelTester):
    def __init__(self, parent, audio_prompt_dim=12, **kwargs):
        super().__init__(parent, **kwargs)
        self.model_class = GradiumPhononTTSForConditionalGeneration if is_torch_available() else None
        # no voice BOS: the vocabulary ends with the voice and latent placeholders and the first audio position
        self.audio_prompt_token_id = self.vocab_size - 3
        self.audio_latent_token_id = self.vocab_size - 2
        self.audio_bos_token_id = self.vocab_size - 1
        self.voice_prefix_ids = []
        # the voice is made of speaker-codec latents, wider than the audio latents
        self.audio_prompt_dim = audio_prompt_dim

    def get_config(self):
        return GradiumPhononTTSConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            latent_dim=self.latent_dim,
            audio_prompt_dim=self.audio_prompt_dim,
            pad_token_id=self.pad_token_id,
            audio_prompt_token_id=self.audio_prompt_token_id,
            audio_latent_token_id=self.audio_latent_token_id,
            audio_bos_token_id=self.audio_bos_token_id,
            flow_head_config=GradiumPhononTTSFlowHeadConfig(
                hidden_size=self.flow_hidden_size,
                num_hidden_layers=self.flow_num_hidden_layers,
                latent_dim=self.latent_dim,
                condition_dim=self.hidden_size,
                flow_type=self.flow_type,
            ),
        )

    def prepare_config_and_inputs(self):
        config, input_ids, attention_mask = super().prepare_config_and_inputs()
        # text ids below the three placeholders
        input_ids[:, :-1] = input_ids[:, :-1] % (self.vocab_size - 4) + 1
        return config, input_ids, attention_mask


@require_torch
class GradiumPhononTTSModelTest(pocket_tts.PocketTTSModelTest):
    all_model_classes = (
        (GradiumPhononTTSModel, GradiumPhononTTSForConditionalGeneration) if is_torch_available() else ()
    )

    def setUp(self):
        self.model_tester = GradiumPhononTTSModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GradiumPhononTTSConfig, has_text_modality=True)
        self.skip_unsupported_generate()

    def test_reverse_loading_mapping(self):
        # the mapping targets the full model's keys (the backbone's under `model.`)
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_conditions_on_audio_latents(self):
        # the conditions are added to every audio latent position, the BOS latent included, not to text or voice
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_with_voice()
        model = GradiumPhononTTSModel(config).to(torch_device).eval()
        with torch.no_grad():
            latents = torch.randn(2, self.model_tester.latent_dim, device=torch_device)
            torch.testing.assert_close(
                model.embed_audio_latents(latents),
                model.audio_latent_proj(latents) + model.get_condition_embeds(torch_device),
            )
            before = model(**inputs_dict).last_hidden_state
            model.config.duration_delta = 0.5
            after = model(**inputs_dict).last_hidden_state
        # only the trailing BOS latent position (and none before it) changes
        torch.testing.assert_close(before[:, :-1], after[:, :-1])
        self.assertFalse(torch.allclose(before[:, -1], after[:, -1]))

    def test_condition_embeds_at_default_values(self):
        # the runtime's defaults: the `num_speakers` row 1, and `duration_delta` 0, whose sinusoidal embedding is
        # cosines of 0 then sines of 0
        config = self.model_tester.get_config()
        model = GradiumPhononTTSModel(config).to(torch_device).eval()
        half_dim = config.duration_delta_embedding_dim // 2
        with torch.no_grad():
            expected = model.num_speakers_proj(model.num_speakers_embed.weight[1])
            expected = expected + model.duration_delta_proj.weight[:, :half_dim].sum(-1)
            torch.testing.assert_close(model.get_condition_embeds(torch_device), expected)


@require_torch
class GradiumPhononTTSIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """
    reproducer (uploads the golden to
    ``hf://buckets/hf-internal-testing/phonon-integration-tests/<case>/expected.safetensors``, holding the waveforms of
    Gradium's runtime, ptts 0.4.0, with the frame counts in ``<case>/metadata.json``):
        https://gist.github.com/eustlb/b54c2ef3b6d0e92b20b23a974fbefad6
    """

    # Tolerance against the goldens from the bucket, on the free-running waveforms: up to 5.3e-4 measured on the CI
    # A10G (2 runs, identical), 7.9e-4 on CPU
    RTOL = 0
    AUDIO_ATOL = 2e-3

    # --- single source of truth, mirrored from the reproducers' common.py ---
    CASES = [
        ("This is Gradium's open-source on-device Text-to-Speech model Phonon", "LMDIYE0Oa377Qh6Z"),  # Easton
        ("Hello world", "NbpkqMVS3CJeq2j8"),  # Zoey; at most 4 words: 3 frames after the end of speech, not 1
    ]
    VOICES = "gradium/phonon-voices"
    VOICES_REVISION = "da6c9ec0d47be8df5c87c730a3aa97187f3229c3"

    checkpoint_name = "gradium/phonon-31m-en"
    bucket = "hf-internal-testing/phonon-integration-tests"

    def setUp(self):
        super().setUp()
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name, device_map=torch_device)

    def _load_voice(self, voice):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(self.VOICES, f"voices/{voice}.safetensors", revision=self.VOICES_REVISION)
        return load_file(path)["speaker_wavs"][0].T.float()

    def _load_expected(self, case):
        from huggingface_hub import download_bucket_files
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            files = [
                (f"{case}/{name}", os.path.join(tmp_dir, name)) for name in ("expected.safetensors", "metadata.json")
            ]
            download_bucket_files(self.bucket, files=files)
            with open(files[1][1], encoding="utf-8") as f:
                return load_file(files[0][1]), json.load(f)

    def _generate(self, cases):
        # the runtime computes in float32 and its reference is greedy (temperature 0)
        model = AutoModelForTextToWaveform.from_pretrained(
            self.checkpoint_name, device_map=torch_device, dtype=torch.float32
        )
        # an empty `audio` item marks the voice, given as precomputed latents
        conversations = [
            [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": text}]}] for text, _ in cases
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs={"audio_prompt_latents": [self._load_voice(voice) for _, voice in cases]},
        ).to(model.device)
        outputs = model.generate(**inputs, do_sample=False)
        return outputs, self.processor.decode(outputs.audio_latents, outputs.audio_latents_mask)

    @slow
    def test_single(self):
        """
        Greedy generation matches the runtime at temperature 0: the end of speech, then the waveform.

        reproducer: https://gist.github.com/eustlb/b54c2ef3b6d0e92b20b23a974fbefad6#file-reproducer-single-py
        """
        expected, metadata = self._load_expected("single")
        outputs, audio = self._generate(self.CASES[:1])

        self.assertEqual(outputs.audio_latents_mask.sum().item(), metadata["sanity"]["num_frames"])
        torch.testing.assert_close(audio[0].cpu(), expected["audio"], rtol=self.RTOL, atol=self.AUDIO_ATOL)

    @slow
    def test_batch(self):
        """
        A left-padded batch of two texts and voices matches the runtime run on each alone: padding is neutral, and the
        short text keeps the runtime's longer tail after the end of speech.

        reproducer: https://gist.github.com/eustlb/b54c2ef3b6d0e92b20b23a974fbefad6#file-reproducer-batch-py
        """
        expected, metadata = self._load_expected("batch")
        outputs, audio = self._generate(self.CASES)

        self.assertListEqual(
            outputs.audio_latents_mask.sum(-1).tolist(), [row["sanity"]["num_frames"] for row in metadata["rows"]]
        )
        for index in range(len(self.CASES)):
            torch.testing.assert_close(
                audio[index].cpu(), expected[f"audio_{index}"], rtol=self.RTOL, atol=self.AUDIO_ATOL
            )
