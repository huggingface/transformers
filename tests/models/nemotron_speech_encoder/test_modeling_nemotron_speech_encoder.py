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
"""Testing suite for the PyTorch NemotronSpeechEncoder model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import NemotronSpeechEncoder, NemotronSpeechEncoderConfig


class NemotronSpeechEncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=64,
        is_training=False,
        num_mel_bins=8,
        subsampling_factor=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        use_qk_norm=True,
        partial_rotary_factor=0.5,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.num_mel_bins = num_mel_bins
        self.subsampling_factor = subsampling_factor
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.use_qk_norm = use_qk_norm
        self.partial_rotary_factor = partial_rotary_factor
        self.encoder_seq_length = seq_length // subsampling_factor
        self.key_length = self.encoder_seq_length

    def get_config(self):
        return NemotronSpeechEncoderConfig(
            num_mel_bins=self.num_mel_bins,
            subsampling_factor=self.subsampling_factor,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            use_qk_norm=self.use_qk_norm,
            partial_rotary_factor=self.partial_rotary_factor,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        # right padding, as the processor produces: the model reads the mask as per-sample lengths
        lengths = torch.randint(self.seq_length // 2, self.seq_length + 1, (self.batch_size,), device=torch_device)
        attention_mask = (torch.arange(self.seq_length, device=torch_device)[None, :] < lengths[:, None]).long()
        return self.get_config(), input_features, attention_mask

    def create_and_check_model(self, config, input_features, attention_mask):
        model = NemotronSpeechEncoder(config=config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.encoder_seq_length, self.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_features": input_features, "attention_mask": attention_mask}


@require_torch
class NemotronSpeechEncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (NemotronSpeechEncoder,) if is_torch_available() else ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = NemotronSpeechEncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NemotronSpeechEncoderConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_partial_rotary_factor(self):
        """Only `partial_rotary_factor` of each head is rotated, as in the original `rotary_fraction`."""
        config = self.model_tester.get_config()
        model = NemotronSpeechEncoder(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self.assertEqual(model.rotary_emb.inv_freq.shape[0], int(head_dim * config.partial_rotary_factor) // 2)

    def test_padding_does_not_leak(self):
        """Valid frames do not depend on the content of the padded frames."""
        config, input_features, _ = self.model_tester.prepare_config_and_inputs()
        model = NemotronSpeechEncoder(config).to(torch_device).eval()
        num_frames = self.model_tester.seq_length // 2 + 1
        attention_mask = torch.zeros_like(input_features[..., 0], dtype=torch.long)
        attention_mask[:, :num_frames] = 1
        corrupted = input_features.clone()
        corrupted[:, num_frames:] = 100.0
        num_valid = (num_frames + config.subsampling_factor - 1) // config.subsampling_factor
        with torch.no_grad():
            expected = model(input_features, attention_mask=attention_mask).last_hidden_state[:, :num_valid]
            result = model(corrupted, attention_mask=attention_mask).last_hidden_state[:, :num_valid]
        torch.testing.assert_close(result, expected)

    @unittest.skip(reason="NemotronSpeechEncoder has no input embedding layer")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="`inputs_embeds` are stacked spectrogram frames, there are no `input_ids` to embed")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="`inputs_embeds` are stacked spectrogram frames, there are no `input_ids` to embed")
    def test_inputs_embeds_matches_input_ids(self):
        pass
