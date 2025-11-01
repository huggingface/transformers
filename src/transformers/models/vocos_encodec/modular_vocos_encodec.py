# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import auto_docstring, can_return_tuple
from ..vocos.modeling_vocos import VocosConvNeXtBlock, VocosModel, VocosOutput, VocosPreTrainedModel
from .configuration_vocos_encodec import VocosEncodecConfig


class VocosEncodecOutput(VocosOutput):
    pass


class VocosEncodecAdaptiveLayerNorm(nn.Module):
    """
    Weight and bias parameters come from a lookup table based on the target bandwidth.
    """

    def __init__(self, config: VocosEncodecConfig):
        super().__init__()
        self.eps = config.layer_norm_eps
        self.hidden_size = config.hidden_size
        adanorm_num_embeddings = len(config.bandwidths)
        self.weight = nn.Parameter(torch.ones(adanorm_num_embeddings, config.hidden_size))
        self.bias = nn.Parameter(torch.zeros(adanorm_num_embeddings, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor, cond_embedding_id: torch.LongTensor):
        hidden_states = F.layer_norm(hidden_states, (self.hidden_size,), weight=None, bias=None, eps=self.eps)
        return hidden_states * self.weight[cond_embedding_id].unsqueeze(0) + self.bias[cond_embedding_id].unsqueeze(0)


class VocosEncodecConvNeXtBlock(VocosConvNeXtBlock):
    def __init__(self, config: VocosEncodecConfig):
        super().__init__(config)
        self.norm = VocosEncodecAdaptiveLayerNorm(config)

    def forward(self, hidden_states: torch.Tensor, bandwidth_id: Optional[torch.LongTensor]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states, cond_embedding_id=bandwidth_id)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.layer_scale_parameter.to(hidden_states.device) * hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = residual.to(hidden_states.device) + hidden_states
        return hidden_states


class VocosEncodecPreTrainedModel(VocosPreTrainedModel):
    config_class = VocosEncodecConfig
    base_model_prefix = "vocos_encodec"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            std = getattr(self.config, "initializer_range", 0.02)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, VocosEncodecAdaptiveLayerNorm):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)


@auto_docstring(
    custom_intro="""
    Vocos model for neural vocoding from EnCodec codes.
    """
)
class VocosEncodecModel(VocosModel):
    def __init__(self, config: VocosEncodecConfig):
        super().__init__(config)

        self.embed = nn.Conv1d(
            config.codebook_dim, config.hidden_size, kernel_size=config.kernel_size, padding=config.padding
        )
        self.norm = VocosEncodecAdaptiveLayerNorm(config)

        # TODO do we keep codebook weights? maybe if they were retrained
        # can be used like this to compute input to `self.embed`:
        # https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/feature_extractors.py#L98
        self.register_buffer("codebook_weights", torch.zeros(config.num_quantizers, config.codebook_dim))
        self._bandwidth_to_id = {bandwidth: id for id, bandwidth in enumerate(config.bandwidths)}

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: Optional[torch.FloatTensor],
        bandwidth: Optional[float] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> VocosEncodecOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_dim, time_dim)`):
            EnCodec neural audio codec features which can be computed either from precomputed EnCodec RVQ codes via
            `processor(codes=codes, bandwidth=1.5)` or from raw audio via `processor(audio=waveform, bandwidth=1.5)`.
            It must be provided with the corresponding EnCodec bandwidth.
        bandwidth (`float`, *optional*):
            Target bandwidth for EnCodec quantizer, e.g. one of [1.5, 3, 6, 12] kbps, to be provided if
            `input_features`is not None.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, time_dim)`, *optional*):
            Padding mask. Not used, but kept so processor outputs can be passed directly.

        Returns:
            `VocosOutput` or tuple `(audio,)`:
            - `audio` of shape (batch_size, time): Reconstructed audio waveform.

        Example:

        ```python
        >>> # Encode audio using EnCodec neural codec and reconstruct from audio from that
        >>> processor = VocosProcessor.from_pretrained("hf-audio/vocos-encodec-24khz")
        >>> model = VocosModel.from_pretrained("hf-audio/vocos-encodec-24khz")

        >>> bandwidth = 6.0
        >>> inputs = processor(audio=audio_sample, bandwidth=bandwidth)
        >>> outputs = model(**inputs)
        >>> reconstructed_audio = outputs.audio

        >>> # Reconstruct audio directly from pre-computed EnCodec quantized codes
        >>> inputs = processor(codes=audio_codes, bandwidth=bandwidth)
        >>> outputs = model(**inputs)
        >>> reconstructed_audio = outputs.audio

        ```
        """
        bandwidth_id = self._bandwidth_to_id[float(bandwidth)]

        hidden_states = self.embed(input_features)

        # Apply initial norm in channel-last format
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states, bandwidth_id)
        hidden_states = hidden_states.transpose(1, 2)

        # Process through ConvNeXt layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, bandwidth_id)
        hidden_states = self.final_layer_norm(hidden_states.transpose(1, 2))

        # Decode back to audio (linear + ISTFT)
        audio = self.decoder(hidden_states)
        return VocosEncodecOutput(audio=audio)


__all__ = ["VocosEncodecModel", "VocosEncodecPreTrainedModel"]
