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
import abc
import functools
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def generate_fourier_features(
    pos, num_bands, max_resolution=(224, 224),
    concat_pos=True, sine_only=False):
  """Generate a Fourier frequency position encoding with linear spacing.
  Args:
    pos: The position of n points in d dimensional space.
      A Torch tensor of shape [n, d].
    num_bands: The number of bands (K) to use.
    max_resolution: The maximum resolution (i.e. the number of pixels per dim).
      A tuple representing resolution for each dimension
    concat_pos: Concatenate the input position encoding to the Fourier features?
    sine_only: Whether to use a single phase (sin) or two (sin/cos) for each
      frequency band.
  Returns:
    embedding: A 1D Torch tensor of shape [n, n_channels]. If concat_pos is True
      and sine_only is False, output dimensions are ordered as:
        [dim_1, dim_2, ..., dim_d,
         sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ...,
         sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d),
         cos(pi*f_1*dim_1), ..., cos(pi*f_K*dim_1), ...,
         cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)],
       where dim_i is pos[:, i] and f_k is the kth frequency band.
  """
  min_freq = 1.0
  # Nyquist frequency at the target resolution:

  freq_bands = torch.stack([
      torch.linspace(min_freq, res / 2, num=num_bands, endpoint=True)
      for res in max_resolution], dim=0)

  # Get frequency bands for each spatial dimension.
  # Output is size [n, d * num_bands]
  per_pos_features = pos[:, :, None] * freq_bands[None, :, :]
  per_pos_features = torch.reshape(per_pos_features,
                                 [-1, np.prod(per_pos_features.shape[1:])])

  if sine_only:
    # Output is size [n, d * num_bands]
    per_pos_features = torch.sin(np.pi * (per_pos_features))
  else:
    # Output is size [n, 2 * d * num_bands]
    per_pos_features = torch.cat(
        [torch.sin(np.pi * per_pos_features),
         torch.cos(np.pi * per_pos_features)], dim=-1)
  # Concatenate the raw input positions.
  if concat_pos:
    # Adds d bands to the encoding.
    per_pos_features = torch.cat([pos, per_pos_features], dim=-1)
  return per_pos_features


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
  """Generate an array of position indices for an N-D input array.
  Args:
    index_dims: The shape of the index dimensions of the input array.
    output_range: The min and max values taken by each input index dimension.
  Returns:
    A Torch tensor of shape [index_dims[0], index_dims[1], .., index_dims[-1], N].
  """
  def _linspace(n_xels_per_dim):
    return torch.linspace(start=output_range[0], end=output_range[1], steps=n_xels_per_dim, dtype=torch.float32)

  dim_ranges = [
      _linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
  array_index_grid = torch.meshgrid(*dim_ranges)

  return torch.stack(array_index_grid, dim=-1)


class PerceiverAbstractPositionEncoding(nn.Module, metaclass=abc.ABCMeta):
  """Perceiver abstract position encoding."""

  @abc.abstractmethod
  def forward(self, batch_size, pos):
    raise NotImplementedError


class PerceiverTrainablePositionEncoding(PerceiverAbstractPositionEncoding):
  """Trainable position encoding."""

  def __init__(self, index_dim, num_channels=128):
    super().__init__()
    self.position_embeddings = nn.Embedding(index_dim, num_channels)

  def forward(self, batch_size, position_ids=None):
    position_embeddings = self.position_embeddings(position_ids)

    if batch_size is not None:
      position_embeddings = position_embeddings.expand(batch_size, -1, -1)
    
    return position_embeddings


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
  """Checks or builds spatial position features (x, y, ...).
  Args:
    pos: None, or an array of position features. If None, position features
      are built. Otherwise, their size is checked.
    index_dims: An iterable giving the spatial/index size of the data to be
      featurized.
    batch_size: The batch size of the data to be featurized.
  Returns:
    An array of position features, of shape [batch_size, prod(index_dims)].
  """
  if pos is None:
    pos = build_linear_positions(index_dims)
    pos = torch.broadcast_to(pos[None], (batch_size,) + pos.shape)
    pos = torch.reshape(pos, [batch_size, np.prod(index_dims), -1])
  else:
    # Just a warning label: you probably don't want your spatial features to
    # have a different spatial layout than your pos coordinate system.
    # But feel free to override if you think it'll work!
    assert pos.shape[-1] == len(index_dims)

  return pos


class PerceiverFourierPositionEncoding(PerceiverAbstractPositionEncoding):
  """Fourier (Sinusoidal) position encoding."""

  def __init__(self, index_dims, num_bands, concat_pos=True,
               max_resolution=None, sine_only=False):
    super().__init__()
    self.num_bands = num_bands
    self.concat_pos = concat_pos
    self.sine_only = sine_only
    self.index_dims = index_dims
    # Use the index dims as the maximum resolution if it's not provided.
    self.max_resolution = max_resolution or index_dims

  def forward(self, batch_size, pos=None):
    pos = _check_or_build_spatial_positions(pos, self.index_dims, batch_size)
    build_ff_fn = functools.partial(
        generate_fourier_features,
        num_bands=self.num_bands,
        max_resolution=self.max_resolution,
        concat_pos=self.concat_pos,
        sine_only=self.sine_only)
    return torch.vmap(build_ff_fn, 0, 0)(pos)


class PerceiverTextPreprocessor(nn.Module):
    """Text preprocessing for Perceiver Encoder."""

    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model)
        self.position_embeddings = nn.Embedding(config.seq_len, config.d_model)
        self.seq_len = config.seq_len

    def forward(self, inputs):

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

    def forward(self, hidden_states, embedding_layer):
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
        in_channels: int = 3,
        out_channels: int = 64,
        conv_after_patching: bool = False,
        conv2d_use_batchnorm: bool = True,
        concat_or_add_pos: str = "concat",
        project_pos_dim=-1,
        **position_encoding_kwargs,
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
        self.position_encoding_type = position_encoding_type
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

        if self.position_encoding_type == "trainable": 
            self.position_embeddings = PerceiverTrainablePositionEncoding(**position_encoding_kwargs)
        elif self.position_encoding_type == "fourier":
            self.position_embeddings = PerceiverFourierPositionEncoding(**position_encoding_kwargs)
        else:
            raise ValueError(f'Unknown position encoding type: {position_encoding_type}.')
        
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
        if self.position_encoding_type == "trainable": 
            position_ids = torch.arange(0, indices)
            pos_enc = self.position_embeddings(batch_size, position_ids)
        elif self.position_encoding_type == "fourier":
            pos_enc = None

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

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        if self.prep_type in ["conv", "patches"]:
            raise NotImplementedError("TODO")
        elif self.prep_type == "conv1x1":
            # map inputs to self.out_channels
            inputs = self.convnet_1x1(inputs)
            #print("Inputs after conv:", inputs[0,:3,:3,:3])
            #print("Sum of inputs after conv:", inputs.sum())

        elif self.prep_type == 'pixels':
            # if requested, downsamples in the crudest way
            if inputs.ndim == 4:
                inputs = inputs[::self.spatial_downsample, ::self.spatial_downsample]
            elif inputs.ndim == 5:
                inputs = inputs[:, ::self.temporal_downsample, :,
                                ::self.spatial_downsample, ::self.spatial_downsample]
            else:
                raise ValueError('Unsupported data format for pixels.')

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos, network_input_is_1d)
        return inputs
        # modality_sizes = None  # Size for each modality, only needed for multimodal
        # return inputs, modality_sizes, inputs_without_pos
