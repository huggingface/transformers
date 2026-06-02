# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass

import torch

from ...generation import GenerationMixin
from ...utils import ModelOutput
from ..parakeet.generation_parakeet import ParakeetRNNTGenerationMixin, ParakeetTDTDecoderCache


class NemotronAsrTDTDecoderCache(ParakeetTDTDecoderCache):
    pass


@dataclass
class NemotronAsrGenerateOutput(ModelOutput):
    """
    Outputs of NemotronAsr RNN-T generation.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Generated token sequences (including blank tokens).
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Encoder attention weights per layer.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Encoder hidden states per layer.
    """

    sequences: torch.LongTensor
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None


class NemotronAsrGenerationMixin(ParakeetRNNTGenerationMixin):
    """Generation mixin for NemotronAsr RNN-T models.

    Identical to [`ParakeetRNNTGenerationMixin`]; only the model-specific cache / encoder-output /
    generate-output classes differ.
    """

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        model_kwargs["decoder_cache"] = NemotronAsrTDTDecoderCache(self.config)

    def prepare_inputs_for_generation(self, input_ids, *args, **kwargs):
        from .modeling_nemotron_asr import NemotronAsrEncoderModelOutput

        # Bypass ParakeetTDTGenerationMixin.prepare_inputs_for_generation (it builds a parakeet output
        # class); go to the base and rebuild with the NemotronAsr encoder-output type.
        model_inputs = GenerationMixin.prepare_inputs_for_generation(self, input_ids, *args, **kwargs)
        encoder_frame_idxs = model_inputs.pop("encoder_frame_idxs").to(
            model_inputs["encoder_outputs"].pooler_output.device
        )

        pooler_output = model_inputs["encoder_outputs"].pooler_output
        batch_size, max_encoder_len = pooler_output.shape[0], pooler_output.shape[1]
        encoder_frame_idxs = encoder_frame_idxs.clamp(max=max_encoder_len - 1)
        model_inputs["encoder_outputs"] = NemotronAsrEncoderModelOutput(
            pooler_output=pooler_output[torch.arange(batch_size), encoder_frame_idxs, None],
        )

        return model_inputs

    def generate(self, inputs=None, generation_config=None, **kwargs):
        self._encoder_finished = None
        self._symbols_at_frame = None
        outputs = GenerationMixin.generate(self, inputs=inputs, generation_config=generation_config, **kwargs)
        del self._encoder_finished, self._symbols_at_frame
        sequences = outputs.sequences if isinstance(outputs, ModelOutput) else outputs
        return NemotronAsrGenerateOutput(sequences=sequences)
