# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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


import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...utils import auto_docstring, can_return_tuple
from ..vocos.modeling_vocos import VocosConvNeXtBlock, VocosModel, VocosOutput, VocosPreTrainedModel
from ..vocos.modeling_vocos import VocosISTFTHead as VocosEncodecISTFTHead
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

    def forward(self, hidden_states: torch.Tensor, bandwidth_id: torch.LongTensor | None) -> torch.Tensor:
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
        if isinstance(module, (VocosEncodecISTFTHead)):
            window = torch.hann_window(module.win_length)
            init.copy_(module.window, window)
        elif isinstance(module, nn.Linear):
            std = getattr(self.config, "initializer_range", 0.02)
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, VocosEncodecAdaptiveLayerNorm):
            if hasattr(module, "bias") and module.bias is not None:
                init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                init.ones_(module.weight)


@auto_docstring(
    custom_intro="""
    Vocos model for neural vocoding reconstructed from EnCodec embedded features.
    """
)
class VocosEncodecModel(VocosModel):
    def __init__(self, config: VocosEncodecConfig):
        super().__init__(config)

        self.embed = nn.Conv1d(
            config.codebook_dim, config.hidden_size, kernel_size=config.kernel_size, padding=config.padding
        )
        self.norm = VocosEncodecAdaptiveLayerNorm(config)

        self.register_buffer("codebook_weights", torch.zeros(config.num_quantizers, config.codebook_dim))
        self._bandwidth_to_id = {bandwidth: id for id, bandwidth in enumerate(config.bandwidths)}

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor | None,
        attention_mask: torch.Tensor | None = None,
        bandwidth: float | None = None,
        **kwargs,
    ) -> VocosEncodecOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_dim, time_dim)`):
            EnCodec neural audio codec features which can be computed either from precomputed EnCodec RVQ codes via
            `processor(codes=codes, bandwidth=1.5)` or from raw audio via `processor(audio=waveform, bandwidth=1.5)`.
            It must be provided with the corresponding EnCodec bandwidth.

        attention_mask (`torch.Tensor` of shape `(batch_size, time)`, *optional*):
            Attention mask indicates which positions contain valid audio vs padding tokens. Not used, returned in the output
            to allow removing padding from reconstructed audios when processing batches with sequences of different lengths (batch_size > 1).

        bandwidth (`float`, *optional*):
            Target bandwidth for EnCodec quantizer, e.g. one of [1.5, 3, 6, 12] kbps, to be provided if
            `input_features`is not None.

        Returns:
            `VocosOutput` or tuple `(audio,)`:
            - `audio` of shape (batch_size, time): Reconstructed audio waveform.
            - `attention_mask` of shape (batch_size, time): atention mask for the reconstructed audio. Not used inside model, used to remove padding from reconstructed audios
                when processing batches with sequences of different lengths (batch_size > 1).


        Example:

        ```python
        >>> from datasets import load_dataset, Audio
        >>> from transformers import VocosEncodecProcessor, VocosEncodecModel

        >>> # Encode audio using `EnCodec` neural codec model into embeddings and reconstruct a higher quality audio from it using `VocosEncodecModel`
        >>> processor = VocosEncodecProcessor.from_pretrained("Manel/vocos-encodec-24khz")
        >>> model = VocosEncodecModel.from_pretrained("Manel/vocos-encodec-24khz")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=24000))
        >>> audios = [x["array"] for x in ds.sort("id")[:3]["audio"]]

        >>> bandwidth = 6.0
        >>> inputs = processor(audio=audios, bandwidth=bandwidth)
        >>> outputs = model(**inputs)
        >>> reconstructed_audio, attention_mask = outputs.audio, outputs.attention_mask

        >>> # Remove padding from reconstructed audios using attention mask
        >>> unpadded_audios = [reconstructed_audio[i][attention_mask[i].bool()].detach().cpu().numpy() for i in range(reconstructed_audio.shape[0])]

        >>> # Reconstruct audio directly from pre-computed EnCodec quantized codes
        >>> inputs = processor(codes=quantized_codes, bandwidth=bandwidth)
        >>> outputs = model(**inputs)
        >>> reconstructed_audio = outputs.audio

        ```
        """

        if bandwidth is None:
            # if model is used without processor to avoid passing without bandwidth
            raise ValueError(
                "VocosEncodecModel requires a `bandwidth` argument, please provide bandwidth in kbps (supported values are [1.5, 3, 6, 12])."
            )

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
        return VocosEncodecOutput(audio=audio, attention_mask=attention_mask)


__all__ = ["VocosEncodecModel", "VocosEncodecPreTrainedModel"]
