# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Sound/Audio model components for multimodal integration.

This module provides the SoundEncoder (wrapping Parakeet from HuggingFace transformers)
and SoundProjection (MLP to project audio embeddings to LLM hidden size).

The Parakeet model in HuggingFace transformers is documented at:
https://huggingface.co/docs/transformers/en/model_doc/parakeet
"""

import torch
import torch.nn as nn

from ...utils import logging
from ..parakeet import ParakeetEncoder, ParakeetEncoderConfig


logger = logging.get_logger(__name__)


class SquaredReLU(nn.Module):
    """Squared ReLU activation function."""

    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 2)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


class SoundProjection(nn.Module):
    """MLP projection from sound encoder hidden size to LLM hidden size.

    Architecture: RMSNorm -> linear1 -> SquaredReLU -> linear2

    This matches the Megatron checkpoint conversion structure:
    - sound_projection.norm.weight
    - sound_projection.linear1.weight
    - sound_projection.linear2.weight
    - sound_projection.linear1.bias (optional)
    - sound_projection.linear2.bias (optional)
    """

    def __init__(
        self,
        sound_hidden_size: int,
        projection_hidden_size: int,
        llm_hidden_size: int,
        bias: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.norm = RMSNorm(sound_hidden_size, eps=eps)
        self.linear1 = nn.Linear(sound_hidden_size, projection_hidden_size, bias=bias)
        self.activation = SquaredReLU()
        self.linear2 = nn.Linear(projection_hidden_size, llm_hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project sound embeddings to LLM embedding space.

        Args:
            hidden_states: Sound encoder output [batch, seq_len, sound_hidden_size]

        Returns:
            Projected embeddings [batch, seq_len, llm_hidden_size]
        """
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class SoundEncoder(nn.Module):
    """Wrapper around the Parakeet encoder from HuggingFace transformers.

    The Parakeet model is an ASR model with a Fast Conformer encoder.
    We use only the encoder portion to extract audio embeddings.

    Checkpoint structure:
    - sound_encoder.encoder.feature_extractor.* -> Feature extraction (mel spectrogram)
    - sound_encoder.encoder.pre_encode.* -> Pre-encoding convolutions
    - sound_encoder.encoder.layers.* -> Conformer layers

    Reference: https://huggingface.co/docs/transformers/en/model_doc/parakeet
    """

    def __init__(self, config=None):
        super().__init__()

        if config is not None:
            # Build from config - handle both dict and config object
            if hasattr(config, "__dict__"):
                # It's a config object, extract relevant params for ParakeetConfig
                config_dict = {
                    "attention_bias": getattr(config, "attention_bias", False),
                    "hidden_size": getattr(config, "hidden_size", 1024),
                    "num_attention_heads": getattr(config, "num_attention_heads", 8),
                    "num_hidden_layers": getattr(config, "num_hidden_layers", 24),
                    "intermediate_size": getattr(config, "intermediate_size", 4096),
                    "conv_kernel_size": getattr(config, "conv_kernel_size", 31),
                    "convolution_bias": getattr(config, "convolution_bias", False),
                    "feat_in": getattr(config, "feat_in", 80),
                    "subsampling_factor": getattr(config, "subsampling_factor", 8),
                    "subsampling_conv_channels": getattr(config, "subsampling_conv_channels", 256),
                    "subsampling_conv_kernel_size": getattr(config, "subsampling_conv_kernel_size", 3),
                    "subsampling_conv_stride": getattr(config, "subsampling_conv_stride", 2),
                    "num_mel_bins": getattr(config, "num_mel_bins", 128),
                    "scale_input": getattr(config, "scale_input", False),
                }
            elif isinstance(config, dict):
                config_dict = config
            else:
                config_dict = {}

            # Create ParakeetConfig with the extracted parameters
            parakeet_config = ParakeetEncoderConfig(**config_dict)
            self.config = parakeet_config
            self.encoder = ParakeetEncoder(parakeet_config)
        else:
            raise ValueError("config must be provided, and ParakeetEncoder must be available in transformers.")

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode audio features.

        Args:
            input_features: Mel spectrogram features [batch, seq_len, feature_dim]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Audio embeddings [batch, encoded_seq_len, hidden_size]
        """
        outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
        )
        # Return the last hidden state
        return outputs.last_hidden_state

    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the encoder."""
        return self.config.hidden_size
