# coding=utf-8
# Copyright 2022 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
""" Flax VQGAN model."""


import math
from functools import partial
from typing import Tuple

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from .configuration_vqgan import VQGANConfig


class Upsample(nn.Module):
    in_channels: int
    with_conv: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(
                self.in_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=((1, 1), (1, 1)),
                dtype=self.dtype,
            )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        if self.with_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


class Downsample(nn.Module):
    in_channels: int
    with_conv: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(
                self.in_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                dtype=self.dtype,
            )

    def __call__(self, hidden_states):
        if self.with_conv:
            pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
            hidden_states = jnp.pad(hidden_states, pad_width=pad)
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = nn.avg_pool(hidden_states, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        return hidden_states


class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: int = None
    use_conv_shortcut: bool = False
    temb_channels: int = 512
    dropout_prob: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.out_channels_ = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv1 = nn.Conv(
            self.out_channels_,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        if self.temb_channels:
            self.temb_proj = nn.Dense(self.out_channels_, dtype=self.dtype)

        self.norm2 = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.conv2 = nn.Conv(
            self.out_channels_,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        if self.in_channels != self.out_channels_:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv(
                    self.out_channels_,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=((1, 1), (1, 1)),
                    dtype=self.dtype,
                )
            else:
                self.nin_shortcut = nn.Conv(
                    self.out_channels_,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="VALID",
                    dtype=self.dtype,
                )

    def __call__(self, hidden_states, temb=None, deterministic: bool = True):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            hidden_states = hidden_states + self.temb_proj(nn.swish(temb))[:, :, None, None]  # TODO: check shapes

        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels_:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return hidden_states + residual


class AttnBlock(nn.Module):
    in_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        conv = partial(
            nn.Conv, self.in_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID", dtype=self.dtype
        )

        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.q, self.k, self.v = conv(), conv(), conv()
        self.proj_out = conv()

    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        # compute attentions
        batch, height, width, channels = query.shape
        query = query.reshape((batch, height * width, channels))
        key = key.reshape((batch, height * width, channels))
        attn_weights = jnp.einsum("...qc,...kc->...qk", query, key)
        attn_weights = attn_weights * (int(channels) ** -0.5)
        attn_weights = nn.softmax(attn_weights, axis=2)

        ## attend to values
        value = value.reshape((batch, height * width, channels))
        hidden_states = jnp.einsum("...kc,...qk->...qc", value, attn_weights)
        hidden_states = hidden_states.reshape((batch, height, width, channels))

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class UpsamplingBlock(nn.Module):
    config: VQGANConfig
    curr_res: int
    block_idx: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.block_idx == self.config.num_resolutions - 1:
            block_in = self.config.ch * self.config.ch_mult[-1]
        else:
            block_in = self.config.ch * self.config.ch_mult[self.block_idx + 1]

        block_out = self.config.ch * self.config.ch_mult[self.block_idx]
        self.temb_ch = 0

        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks + 1):
            res_blocks.append(
                ResnetBlock(
                    block_in, block_out, temb_channels=self.temb_ch, dropout_prob=self.config.dropout, dtype=self.dtype
                )
            )
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in, dtype=self.dtype))

        self.block = res_blocks
        self.attn = attn_blocks

        self.upsample = None
        if self.block_idx != 0:
            self.upsample = Upsample(block_in, self.config.resamp_with_conv, dtype=self.dtype)

    def __call__(self, hidden_states, temb=None, deterministic: bool = True):
        for res_block in self.block:
            hidden_states = res_block(hidden_states, temb, deterministic=deterministic)
            for attn_block in self.attn:
                hidden_states = attn_block(hidden_states)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class DownsamplingBlock(nn.Module):
    config: VQGANConfig
    curr_res: int
    block_idx: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        in_ch_mult = (1,) + tuple(self.config.ch_mult)
        block_in = self.config.ch * in_ch_mult[self.block_idx]
        block_out = self.config.ch * self.config.ch_mult[self.block_idx]
        self.temb_ch = 0

        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(
                ResnetBlock(
                    block_in, block_out, temb_channels=self.temb_ch, dropout_prob=self.config.dropout, dtype=self.dtype
                )
            )
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in, dtype=self.dtype))

        self.block = res_blocks
        self.attn = attn_blocks

        self.downsample = None
        if self.block_idx != self.config.num_resolutions - 1:
            self.downsample = Downsample(block_in, self.config.resamp_with_conv, dtype=self.dtype)

    def __call__(self, hidden_states, temb=None, deterministic: bool = True):
        for res_block in self.block:
            hidden_states = res_block(hidden_states, temb, deterministic=deterministic)
            for attn_block in self.attn:
                hidden_states = attn_block(hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        return hidden_states


class MidBlock(nn.Module):
    in_channels: int
    temb_channels: int
    dropout: float
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.block_1 = ResnetBlock(
            self.in_channels,
            self.in_channels,
            temb_channels=self.temb_channels,
            dropout_prob=self.dropout,
            dtype=self.dtype,
        )
        self.attn_1 = AttnBlock(self.in_channels, dtype=self.dtype)
        self.block_2 = ResnetBlock(
            self.in_channels,
            self.in_channels,
            temb_channels=self.temb_channels,
            dropout_prob=self.dropout,
            dtype=self.dtype,
        )

    def __call__(self, hidden_states, temb=None, deterministic: bool = True):
        hidden_states = self.block_1(hidden_states, temb, deterministic=deterministic)
        hidden_states = self.attn_1(hidden_states)
        hidden_states = self.block_2(hidden_states, temb, deterministic=deterministic)
        return hidden_states


class Encoder(nn.Module):
    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.temb_ch = 0

        # downsampling
        self.conv_in = nn.Conv(
            self.config.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        curr_res = self.config.resolution
        downsample_blocks = []
        for i_level in range(self.config.num_resolutions):
            downsample_blocks.append(DownsamplingBlock(self.config, curr_res, block_idx=i_level, dtype=self.dtype))

            if i_level != self.config.num_resolutions - 1:
                curr_res = curr_res // 2
        self.down = downsample_blocks

        # middle
        mid_channels = self.config.ch * self.config.ch_mult[-1]
        self.mid = MidBlock(mid_channels, self.temb_ch, self.config.dropout, dtype=self.dtype)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv_out = nn.Conv(
            2 * self.config.z_channels if self.config.double_z else self.config.z_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, pixel_values, deterministic: bool = True):
        # timestep embedding
        temb = None

        # downsampling
        hidden_states = self.conv_in(pixel_values)
        for block in self.down:
            hidden_states = block(hidden_states, temb, deterministic=deterministic)

        # middle
        hidden_states = self.mid(hidden_states, temb, deterministic=deterministic)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.temb_ch = 0

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = self.config.ch * self.config.ch_mult[self.config.num_resolutions - 1]
        curr_res = self.config.resolution // 2 ** (self.config.num_resolutions - 1)
        self.z_shape = (1, self.config.z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv(
            block_in,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # middle
        self.mid = MidBlock(block_in, self.temb_ch, self.config.dropout, dtype=self.dtype)

        # upsampling
        upsample_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            upsample_blocks.append(UpsamplingBlock(self.config, curr_res, block_idx=i_level, dtype=self.dtype))
            if i_level != 0:
                curr_res = curr_res * 2
        self.up = list(reversed(upsample_blocks))  # reverse to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv_out = nn.Conv(
            self.config.out_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states, deterministic: bool = True):
        # timestep embedding
        temb = None

        # z to block_in
        hidden_states = self.conv_in(hidden_states)

        # middle
        hidden_states = self.mid(hidden_states, temb, deterministic=deterministic)

        # upsampling
        for block in reversed(self.up):
            hidden_states = block(hidden_states, temb, deterministic=deterministic)

        # end
        if self.config.give_pre_end:
            return hidden_states

        hidden_states = self.norm_out(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________ Discretization bottleneck part of the VQ-VAE. Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedding = nn.Embed(self.config.n_embed, self.config.embed_dim, dtype=self.dtype)  # TODO: init

    def __call__(self, hidden_states):
        """
        Inputs the output of the encoder network z and maps it to a discrete one-hot vector that is the index of the
        closest embedding vector e_j z (continuous) -> z_q (discrete) z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        #  flatten
        hidden_states_flattended = hidden_states.reshape((-1, self.config.embed_dim))

        # dummy op to init the weights, so we can access them below
        self.embedding(jnp.ones((1, 1), dtype="i4"))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        emb_weights = self.variables["params"]["embedding"]["embedding"]
        distance = (
            jnp.sum(hidden_states_flattended**2, axis=1, keepdims=True)
            + jnp.sum(emb_weights**2, axis=1)
            - 2 * jnp.dot(hidden_states_flattended, emb_weights.T)
        )

        # get quantized latent vectors
        min_encoding_indices = jnp.argmin(distance, axis=1)
        z_q = self.embedding(min_encoding_indices).reshape(hidden_states.shape)

        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(hidden_states.shape[0], -1)

        # compute the codebook_loss (q_loss) outside the model
        # here we return the embeddings and indices
        return z_q, min_encoding_indices

    def get_codebook_entry(self, indices, shape=None):
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        batch, num_tokens = indices.shape
        z_q = self.embedding(indices)
        z_q = z_q.reshape(batch, int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens)), -1)
        return z_q


class VQModule(nn.Module):
    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.encoder = Encoder(self.config, dtype=self.dtype)
        self.decoder = Decoder(self.config, dtype=self.dtype)
        self.quantize = VectorQuantizer(self.config, dtype=self.dtype)
        self.quant_conv = nn.Conv(
            self.config.embed_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )
        self.post_quant_conv = nn.Conv(
            self.config.z_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )

    def encode(self, pixel_values, deterministic: bool = True):
        hidden_states = self.encoder(pixel_values, deterministic=deterministic)
        hidden_states = self.quant_conv(hidden_states)
        quant_states, indices = self.quantize(hidden_states)
        return quant_states, indices

    def decode(self, hidden_states, deterministic: bool = True):
        hidden_states = self.post_quant_conv(hidden_states)
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)
        return hidden_states

    def decode_code(self, code_b):
        hidden_states = self.quantize.get_codebook_entry(code_b)
        hidden_states = self.decode(hidden_states)
        return hidden_states

    def __call__(self, pixel_values, deterministic: bool = True):
        quant_states, indices = self.encode(pixel_values, deterministic)
        hidden_states = self.decode(quant_states, deterministic)
        return hidden_states, indices


class VQGANPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VQGANConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: VQGANConfig,
        input_shape: Tuple = (1, 256, 256, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=jnp.float32)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, pixel_values)["params"]

    def encode(self, pixel_values, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train: bool = False):
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params}, jnp.array(pixel_values), not train, rngs=rngs, method=self.module.encode
        )

    def decode(self, hidden_states, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train: bool = False):
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(hidden_states),
            not train,
            rngs=rngs,
            method=self.module.decode,
        )

    def decode_code(self, indices, params: dict = None):
        return self.module.apply(
            {"params": params or self.params}, jnp.array(indices, dtype="i4"), method=self.module.decode_code
        )

    def __call__(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
    ):
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values),
            not train,
            rngs=rngs,
        )


class VQModel(VQGANPreTrainedModel):
    module_class = VQModule
