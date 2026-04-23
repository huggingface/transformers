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
"""Testing suite for the IBM Granite Speech Plus model."""

import unittest

from transformers import (
    GraniteSpeechPlusConfig,
    GraniteSpeechPlusForConditionalGeneration,
)
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ..granite_speech.test_modeling_granite_speech import (
    GraniteSpeechForConditionalGenerationModelTest as _GraniteSpeechModelTestBase,
    GraniteSpeechForConditionalGenerationModelTester as _GraniteSpeechModelTesterBase,
)


if is_torch_available():
    import torch


class GraniteSpeechPlusForConditionalGenerationModelTester(_GraniteSpeechModelTesterBase):
    """
    Plus variant that exercises the ``encoder_hidden_layers`` concat path. The projector's
    ``encoder_hidden_size`` is scaled to match ``encoder_config.hidden_dim * (len(encoder_hidden_layers) + 1)``.
    """

    def __init__(self, parent, encoder_hidden_layers=(0,), **kwargs):
        projector_config = kwargs.pop(
            "projector_config",
            {
                "attention_probs_dropout_prob": 0.1,
                "cross_attention_frequency": 1,
                "encoder_hidden_size": 64,  # 32 (hidden_dim) * (1 intermediate + 1 last) = 64
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 32,
                "initializer_range": 0.02,
                "intermediate_size": 256,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 2048,
                "model_type": "blip_2_qformer",
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "use_qformer_text_input": False,
                "vocab_size": 30522,
            },
        )
        super().__init__(parent=parent, projector_config=projector_config, **kwargs)
        self.encoder_hidden_layers = list(encoder_hidden_layers)

    def get_config(self):
        return GraniteSpeechPlusConfig(
            encoder_config=self.encoder_config,
            text_config=self.text_config,
            projector_config=self.projector_config,
            audio_token_index=self.audio_token_index,
            tie_word_embeddings=self.tie_word_embeddings,
            initializer_range=self.initializer_range,
            has_lora_adapter=self.has_lora_adapter,
            encoder_hidden_layers=self.encoder_hidden_layers,
        )


@require_torch
class GraniteSpeechPlusForConditionalGenerationModelTest(_GraniteSpeechModelTestBase):
    """
    Model tester for `GraniteSpeechPlusForConditionalGeneration`.
    """

    all_model_classes = (GraniteSpeechPlusForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"any-to-any": GraniteSpeechPlusForConditionalGeneration} if is_torch_available() else {}

    def setUp(self):
        self.model_tester = GraniteSpeechPlusForConditionalGenerationModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=GraniteSpeechPlusConfig,
            has_text_modality=False,
        )

    def test_encoder_hidden_layers_concat_shape(self):
        """With ``encoder_hidden_layers`` set, get_audio_features concatenates the selected intermediate
        hidden states with the final hidden state before the projector."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteSpeechPlusForConditionalGeneration(config).to(
            self.model_tester.parent.device if hasattr(self.model_tester.parent, "device") else "cpu"
        )
        model.eval()
        with torch.no_grad():
            out = model.get_audio_features(inputs_dict["input_features"].to(next(model.parameters()).device))
        self.assertEqual(out.pooler_output.shape[0], inputs_dict["input_features"].shape[0])


if __name__ == "__main__":
    unittest.main()
