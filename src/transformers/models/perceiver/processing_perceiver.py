# coding=utf-8
# Copyright 2021 Deepmind and The HuggingFace Inc. team.
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
"""
IO pre- and post-processor classes for Perceiver.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class PerceiverTextPreprocessor(nn.Module):
    """Text preprocessing for Perceiver Encoder."""

    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model)
        self.position_embeddings = nn.Embedding(config.seq_len, config.d_model)
        self.seq_len = config.seq_len

    def __call__(self, inputs):

        embeddings = self.embeddings(inputs)
        position_ids = torch.arange(0, self.seq_len)
        embeddings = embeddings + self.position_embeddings(position_ids)

        return embeddings


class PerceiverTextPostprocessor(nn.Module):
    """Module to decode embeddings."""

    def __init__(self, config):
        """Constructs the module."""
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.bias = nn.Parameter(torch.zeros(self.vocab_size))

    def __call__(self, hidden_states, embedding_layer):
        batch_size, seq_len, d_model = hidden_states.shape
        output = torch.matmul(hidden_states.reshape([-1, d_model]), embedding_layer.weight.T)  # Flatten batch dim
        output = output + self.bias

        return output.reshape([batch_size, seq_len, self.vocab_size])


class PerceiverImagePreprocessor(nn.Module):
    """Image preprocessing for Perceiver Encoder."""

    def __init__(
        self,
        config,
        prep_type="conv",
        spatial_downsample: int = 4,
        temporal_downsample: int = 1,
        position_encoding_type: str = "fourier",
        n_extra_pos_mlp: int = 0,
        n_positions: int = 50176,
        in_channels: int = 3,
        out_channels: int = 64,
        conv_after_patching: bool = False,
        conv2d_use_batchnorm: bool = True,
        concat_or_add_pos: str = "concat",
        project_pos_dim=-1,
    ):
        super().__init__()
        self.config = config

        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError("Invalid prep_type!")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample
        self.concat_or_add_pos = concat_or_add_pos
        self.conv_after_patching = conv_after_patching
        self.out_channels = out_channels

        if self.prep_type in ["conv", "patches", "pixels"]:
            raise NotImplementedError(f"Preparation type {prep_type} is not yet supported")
        elif self.prep_type == "conv1x1":
            assert temporal_downsample == 1, "conv1x1 does not downsample in time."
            self.convnet_1x1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=[1, 1],
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=(spatial_downsample, spatial_downsample),
            )

        self.position_embeddings = nn.Embedding(n_positions, self.out_channels)

        # Stack MLPs to get a deeper positional embedding.
        if n_extra_pos_mlp > 0:
            raise NotImplementedError("Stacking MLPs is not yet supported")

        self.positions_projection = None
        if project_pos_dim > 0:
            self.positions_projection = nn.Linear(self.out_channels, project_pos_dim)

    def _build_network_inputs(self, inputs: torch.Tensor, pos: torch.Tensor, network_input_is_1d: bool = True):
        """Construct the final input, including position encoding."""
        # inputs have shape (batch_size, num_channels, height, width)
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[2:]
        indices = np.prod(index_dims)
        
        # Reshape input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            # Move axes from (batch_size, num_channels, height, width) to (batch_size, height, width, num_channels)
            # as the original implementation expects the channels to be as last dimension before flattening
            inputs = torch.moveaxis(inputs, 1, -1)
            inputs = torch.reshape(inputs, [batch_size, indices, -1])

        #print("Shape of inputs:", inputs.shape)
        #print("First elements of inputs:", inputs[0,:3,:3])
        #print("Sum of inputs before adding position encodings:", inputs.sum())
        
        # Construct the position encoding.
        position_ids = torch.arange(0, indices)
        pos_enc = self.position_embeddings(position_ids)
        pos_enc = pos_enc.expand(batch_size, -1, -1)

        #print("Shape of position encodings before projection:", pos_enc.shape)
        #print("First elements of position encodings before projection:", pos_enc[0,:3,:3])
        #print("Sum of position encodings before projection:", pos_enc.sum())

        #print("Shape of weights of position projector:", self.positions_projection.weight.shape)
        #print("First elements of weights of position projector:", self.positions_projection.weight[:3,:3])
        
        # Optionally project them to a target dimension.
        if self.positions_projection is not None:
            pos_enc = self.positions_projection(pos_enc)

        #print("Shape of position encodings after projection:", pos_enc.shape)
        #print("First elements of position encodings after projection:", pos_enc[0,:3,:3])
        #print("Sum of position encodings after projection:", pos_enc.sum())

        if not network_input_is_1d:
            # Reshape pos to match the input feature shape
            # if the network takes non-1D inputs
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])

        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        #print("Inputs with position encodings:", inputs_with_pos[0,:3,:3])
        #print("Sum of inputs with position encodings:", inputs_with_pos.sum())
        #print("Inputs without position encodings:", inputs[0,:3,:3])
        #print("Sum of inputs without position encodings:", inputs.sum())
    
        return inputs_with_pos, inputs

    def __call__(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        if self.prep_type in ["conv", "patches", "pixels"]:
            raise NotImplementedError("TODO")
        elif self.prep_type == "conv1x1":
            # map inputs to self.out_channels
            inputs = self.convnet_1x1(inputs)
            #print("Inputs after conv:", inputs[0,:3,:3,:3])
            #print("Sum of inputs after conv:", inputs.sum())

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos, network_input_is_1d)
        return inputs
        # modality_sizes = None  # Size for each modality, only needed for multimodal
        # return inputs, modality_sizes, inputs_without_pos
