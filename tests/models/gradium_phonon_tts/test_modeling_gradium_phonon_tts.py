# Copyright 2026 Gradium and The HuggingFace Inc. team. All rights reserved.
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

from transformers import GradiumPhononTTSConfig, GradiumPhononTTSFlowHeadConfig
from transformers.testing_utils import is_torch_available, require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ..pocket_tts import test_modeling_pocket_tts as pocket_tts


if is_torch_available():
    import torch

    from transformers import GradiumPhononTTSForConditionalGeneration, GradiumPhononTTSModel


class GradiumPhononTTSModelTester(pocket_tts.PocketTTSModelTester):
    def __init__(self, parent, audio_prompt_dim=12, **kwargs):
        super().__init__(parent, **kwargs)
        self.model_class = GradiumPhononTTSForConditionalGeneration if is_torch_available() else None
        # no voice BOS: the vocabulary ends with the voice and latent placeholders
        self.audio_prompt_token_id = self.vocab_size - 2
        self.audio_latent_token_id = self.vocab_size - 1
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
        # text ids below the two placeholders
        input_ids[:, :-1] = input_ids[:, :-1] % (self.vocab_size - 3) + 1
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

    def test_speaker_embedding_on_audio_latents(self):
        # the speaker vector is added to every audio latent position, the BOS latent included, not to text or voice
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_with_voice()
        model = GradiumPhononTTSModel(config).to(torch_device).eval()
        with torch.no_grad():
            latents = torch.randn(2, self.model_tester.latent_dim, device=torch_device)
            model.num_speakers_embedding.normal_()
            torch.testing.assert_close(
                model.embed_audio_latents(latents),
                model.audio_latent_proj(latents) + model.num_speakers_embedding,
            )
            before = model(**inputs_dict).last_hidden_state
            model.num_speakers_embedding.zero_()
            after = model(**inputs_dict).last_hidden_state
        # only the trailing BOS latent position (and none before it) changes
        torch.testing.assert_close(before[:, :-1], after[:, :-1])
        self.assertFalse(torch.allclose(before[:, -1], after[:, -1]))
