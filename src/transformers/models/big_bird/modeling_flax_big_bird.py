# coding=utf-8
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
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

from typing import Callable, Optional, Tuple

import numpy as np

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla
from flax.core.frozen_dict import FrozenDict
from flax.linen.attention import dot_product_attention_weights
from jax import lax

from ...file_utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import logging
from .configuration_big_bird import BigBirdConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/bigbird-roberta-base"
_CONFIG_FOR_DOC = "BigBirdConfig"
_TOKENIZER_FOR_DOC = "BigBirdTokenizer"


@flax.struct.dataclass
class FlaxBigBirdForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.BigBirdForPreTraining`.

    Args:
        prediction_logits (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(jax_xla.DeviceArray)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jax_xla.DeviceArray` (one for the output of the embeddings + one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jax_xla.DeviceArray)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jax_xla.DeviceArray` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    prediction_logits: jax_xla.DeviceArray = None
    seq_relationship_logits: jax_xla.DeviceArray = None
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = None


@flax.struct.dataclass
class FlaxBigBirdForQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        start_logits (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        pooled_output (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, hidden_size)`):
            pooled_output returned by FlaxBigBirdModel.
        hidden_states (:obj:`tuple(jax_xla.DeviceArray)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jax_xla.DeviceArray` (one for the output of the embeddings + one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jax_xla.DeviceArray)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jax_xla.DeviceArray` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    start_logits: jax_xla.DeviceArray = None
    end_logits: jax_xla.DeviceArray = None
    pooled_output: jax_xla.DeviceArray = None
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = None


BIG_BIRD_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.FlaxPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading, saving and converting weights from
    PyTorch models)

    This model is also a Flax Linen `flax.linen.Module
    <https://flax.readthedocs.io/en/latest/flax.linen.html#module>`__ subclass. Use it as a regular Flax linen Module
    and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - `Just-In-Time (JIT) compilation <https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit>`__
    - `Automatic Differentiation <https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation>`__
    - `Vectorization <https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap>`__
    - `Parallelization <https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap>`__

    Parameters:
        config (:class:`~transformers.BigBirdConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.FlaxPreTrainedModel.from_pretrained` method to load the
            model weights.
"""

BIG_BIRD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`numpy.ndarray` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BigBirdTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`numpy.ndarray` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`numpy.ndarray` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`numpy.ndarray` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.

"""


class FlaxBigBirdEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # Copied from transformers.models.bert.modeling_flax_bert.FlaxBertEmbeddings.setup
    def setup(self):
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # Embed
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        if self.config.rescale_embeddings:
            inputs_embeds *= self.config.hidden_size ** 0.5

        # Sum all embeddings
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention with Bert->BigBird
class FlaxBigBirdSelfAttention(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`\
                    : {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
        )

    def __call__(self, hidden_states, attention_mask, deterministic=True, output_attentions: bool = False):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, -1e10).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxBigBirdBlockSparseAttention(nn.Module):
    config: BigBirdConfig
    block_sparse_seed: int = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            use_bias=self.config.use_bias,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            use_bias=self.config.use_bias,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            use_bias=self.config.use_bias,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
        )

    @staticmethod
    def transpose_for_scores(x, n_heads, head_size):
        new_x_shape = x.shape[:-1] + (n_heads, head_size)
        x = x.reshape(*new_x_shape)
        return jnp.transpose(x, axes=(0, 2, 1, 3))

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic=True,
        output_attentions=False,
    ):
        n_heads = self.config.num_attention_heads
        head_size = self.config.hidden_size // n_heads

        blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(
            attention_mask, self.config.block_size
        )

        query_layer = self.transpose_for_scores(self.query(hidden_states), n_heads, head_size)
        key_layer = self.transpose_for_scores(self.key(hidden_states), n_heads, head_size)
        value_layer = self.transpose_for_scores(self.value(hidden_states), n_heads, head_size)

        attn_output, attn_weights = self.bigbird_block_sparse_attention(
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            blocked_encoder_mask,
            blocked_encoder_mask,
            n_heads,
            head_size,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=output_attentions,
        )

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

    @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask, block_size: int):

        batch_size, seq_length = attention_mask.shape
        assert (
            seq_length % block_size == 0
        ), f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block size is {block_size}."

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            Create 3D attention mask from a 2D tensor mask.

            Args:
                from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].

            Returns:
                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
                3*to_block_size].
            """
            exp_blocked_to_pad = jnp.concatenate(
                [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], axis=2
            )
            band_mask = jnp.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask = jnp.expand_dims(band_mask, 1)
            return band_mask

        blocked_encoder_mask = attention_mask.reshape(batch_size, seq_length // block_size, block_size)
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

        from_mask = attention_mask.reshape(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.reshape(batch_size, 1, 1, seq_length)

        return blocked_encoder_mask, band_mask, from_mask, to_mask

    def bigbird_block_sparse_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        n_heads,
        head_size,
        plan_from_length=None,
        plan_num_rand_blocks=None,
        output_attentions=None,
    ):
        # BigBird block-sparse attention as suggested in paper

        # ITC:
        #     global tokens: 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # ETC:
        #     global tokens: extra_globals_tokens + 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # Note:
        #     1) Currently, ETC is not supported.
        #     2) Window size is fixed to 3 blocks & it can be changed only by
        #     changing `block_size`.
        #     3) Number of global blocks are fixed (2 blocks here) & global tokens can be
        #     controlled only by `block_size`.

        # attention is calculated separately for q[0], q[1], q[2:-2], q[-2], q[-1] in order to use special trick of
        # shifting tokens (for calculating sliding attention). hence following code can be divided into 5 parts.

        bsz, _, from_seq_len, _ = query_layer.shape
        to_seq_len = key_layer.shape[2]
        from_block_size = to_block_size = self.config.block_size

        assert from_seq_len % from_block_size == 0, "Query sided sequence length must be multiple of block size"
        assert to_seq_len % to_block_size == 0, "Key/Value sided sequence length must be multiple of block size"
        if from_seq_len // from_block_size != to_seq_len // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        n_rand_blocks = self.config.num_random_blocks
        rsqrt_d = 1 / jnp.sqrt(head_size)
        attn_mask_penalty = -10000.0

        np.random.seed(self.block_sparse_seed)
        if from_seq_len in [1024, 3072, 4096]:  # old plans used in paper
            max_seqlen = self.config.max_position_embeddings
            rand_attn = [
                self._bigbird_block_rand_mask(
                    max_seqlen, max_seqlen, from_block_size, to_block_size, n_rand_blocks, last_idx=1024
                )[: (from_seq_len // from_block_size - 2)]
                for _ in range(n_heads)
            ]
        else:
            if plan_from_length is None:
                plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(
                    from_seq_len, from_block_size, n_rand_blocks
                )

            rand_attn = self._bigbird_block_rand_mask_with_head(
                from_seq_length=from_seq_len,
                to_seq_length=to_seq_len,
                from_block_size=from_block_size,
                to_block_size=to_block_size,
                num_heads=n_heads,
                plan_from_length=plan_from_length,
                plan_num_rand_blocks=plan_num_rand_blocks,
            )

        rand_attn = jnp.stack(rand_attn, axis=0)
        rand_attn = jnp.broadcast_to(rand_attn, (bsz,) + rand_attn.shape)

        rand_mask = self._create_rand_mask_from_inputs(
            from_blocked_mask, to_blocked_mask, rand_attn, n_heads, n_rand_blocks, bsz, from_seq_len, from_block_size
        )

        blocked_query_matrix = query_layer.reshape(bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1)
        blocked_key_matrix = key_layer.reshape(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        blocked_value_matrix = value_layer.reshape(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)

        shape = (bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1)
        gathered_key = self.jax_gather(blocked_key_matrix, rand_attn, batch_dims=2).reshape(*shape)
        gathered_value = self.jax_gather(blocked_value_matrix, rand_attn, batch_dims=2).reshape(*shape)

        # 1st PART
        # 1st block (global block) attention scores
        # q[0] x (k[0], k[1], k[2], k[3], k[4] .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        first_product = jnp.einsum("bhqd,bhkd->bhqk", blocked_query_matrix[:, :, 0], key_layer)

        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = jax.nn.softmax(first_product, axis=-1)  # [bsz, n_heads, from_block_size, to_seq_len]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        first_context_layer = jnp.einsum("bhqk,bhkd->bhqd", first_attn_weights, value_layer)
        first_context_layer = jnp.expand_dims(first_context_layer, 2)

        # 2nd PART
        # 2nd block attention scores
        # q[1] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> 2nd, 3rd blocks
        # global key blocks -> 1st block

        second_key_mat = jnp.concatenate(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1],
                blocked_key_matrix[:, :, 2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, 0],
            ],
            axis=2,
        )  # [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]
        second_value_mat = jnp.concatenate(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1],
                blocked_value_matrix[:, :, 2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, 0],
            ],
            axis=2,
        )  # [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]
        # ==> [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]
        second_product = jnp.einsum("bhqd,bhkd->bhqk", blocked_query_matrix[:, :, 1], second_key_mat)
        second_seq_pad = jnp.concatenate(
            [
                to_mask[:, :, :, : 3 * to_block_size],
                to_mask[:, :, :, -to_block_size:],
                jnp.ones([bsz, 1, 1, n_rand_blocks * to_block_size], dtype=to_mask.dtype),
            ],
            axis=3,
        )
        second_rand_pad = jnp.concatenate(
            [
                jnp.ones([bsz, n_heads, from_block_size, 4 * to_block_size], dtype=rand_mask.dtype),
                rand_mask[:, :, 0],
            ],
            axis=3,
        )
        second_product = second_product * rsqrt_d
        second_product += (1.0 - jnp.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = jax.nn.softmax(
            second_product, axis=-1
        )  # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+r)*to_block_size] x [bsz, n_heads, (4+r)*to_block_size, -1]
        #  ==> [bsz, n_heads, from_block_size, -1]
        second_context_layer = jnp.einsum("bhqk,bhkd->bhqd", second_attn_weights, second_value_mat)
        second_context_layer = jnp.expand_dims(second_context_layer, 2)

        # 3rd PART
        # Middle blocks attention scores
        # q[-2:2] x (sliding_keys, random_keys, global_keys)
        # sliding attn is calculated using special trick of shifting tokens as discussed in paper
        # random keys are generated by taking random indices as per `rand_attn`
        # global keys -> 1st & last block

        exp_blocked_key_matrix = jnp.concatenate(
            [blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], axis=3
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        exp_blocked_value_matrix = jnp.concatenate(
            [blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]],
            axis=3,
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]

        # sliding attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [b, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        inner_band_product = jnp.einsum("bhlqd,bhlkd->bhlqk", middle_query_matrix, exp_blocked_key_matrix)
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, 3*to_block_size]
        inner_band_product = inner_band_product * rsqrt_d

        # randn attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]
        # x [bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, -1]
        rand_band_product = jnp.einsum("bhlqd,bhlkd->bhlqk", middle_query_matrix, gathered_key[:, :, 1:-1])
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, n_rand_blocks*to_block_size]
        rand_band_product = rand_band_product * rsqrt_d

        # Including 1st block (since it's global)
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1]
        #  ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        first_band_product = jnp.einsum("bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0])
        first_band_product = first_band_product * rsqrt_d

        # Including last block (since it's global)
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1]
        #  ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        last_band_product = jnp.einsum("bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1])
        last_band_product = last_band_product * rsqrt_d

        # masking padded tokens
        inner_band_product += (1.0 - band_mask) * attn_mask_penalty
        first_band_product += (1.0 - jnp.expand_dims(to_mask[:, :, :, :to_block_size], 3)) * attn_mask_penalty
        last_band_product += (1.0 - jnp.expand_dims(to_mask[:, :, :, -to_block_size:], 3)) * attn_mask_penalty
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty

        # completing attention scores matrix for all q[-2:2]
        band_product = jnp.concatenate(
            [first_band_product, inner_band_product, rand_band_product, last_band_product], axis=-1
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+n_rand_blocks)*to_block_size]

        # safely doing softmax since attention matrix is completed
        attn_weights = jax.nn.softmax(
            band_product, axis=-1
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+n_rand_blocks)*to_block_size]

        # contribution of sliding keys
        # [bsz, n_heads, m//from_block_size-4, from_block_size, 3*to_block_size]
        # x [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        context_layer = jnp.einsum(
            "bhlqk,bhlkd->bhlqd", attn_weights[:, :, :, :, to_block_size : 4 * to_block_size], exp_blocked_value_matrix
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # adding contribution of random keys
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, n_rand_blocks*to_block_size]
        # x [bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, -1]
        context_layer += jnp.einsum(
            "bhlqk,bhlkd->bhlqd",
            attn_weights[:, :, :, :, 4 * to_block_size : -to_block_size],
            gathered_value[:, :, 1:-1],
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # adding contribution of global keys
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1]
        #  ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]
        context_layer += jnp.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, :to_block_size], blocked_value_matrix[:, :, 0]
        )
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1]
        # ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]
        context_layer += jnp.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, -to_block_size:], blocked_value_matrix[:, :, -1]
        )

        # 4th PART
        # last 2nd token attention scores
        # q[-2] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> last 3 blocks
        # global key block -> 1st block
        # random key block -> based on indices stored in `randn_attn`

        second_last_key_mat = jnp.concatenate(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, -3],
                blocked_key_matrix[:, :, -2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, -1],
            ],
            axis=2,
        )  # [bsz, n_heads, (4+n_random_blocks)*to_block_size, -1]
        second_last_value_mat = jnp.concatenate(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, -3],
                blocked_value_matrix[:, :, -2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, -1],
            ],
            axis=2,
        )  # [bsz, n_heads, (4+r)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]
        # ==> [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]
        second_last_product = jnp.einsum("bhqd,bhkd->bhqk", blocked_query_matrix[:, :, -2], second_last_key_mat)
        second_last_seq_pad = jnp.concatenate(
            [
                to_mask[:, :, :, :to_block_size],
                to_mask[:, :, :, -3 * to_block_size :],
                jnp.ones([bsz, 1, 1, n_rand_blocks * to_block_size], dtype=to_mask.dtype),
            ],
            axis=3,
        )
        second_last_rand_pad = jnp.concatenate(
            [
                jnp.ones([bsz, n_heads, from_block_size, 4 * to_block_size], dtype=rand_mask.dtype),
                rand_mask[:, :, -1],
            ],
            axis=3,
        )
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - jnp.minimum(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = jax.nn.softmax(
            second_last_product, axis=-1
        )  # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]
        # ==> [bsz, n_heads, from_block_size, -1]
        second_last_context_layer = jnp.einsum("bhqk,bhkd->bhqd", second_last_attn_weights, second_last_value_mat)
        second_last_context_layer = jnp.expand_dims(second_last_context_layer, 2)

        # 5th PART
        # last block (global) attention scores
        # q[-1] x (k[0], k[1], k[2], k[3], .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        last_product = jnp.einsum("bhqd,bhkd->bhqk", blocked_query_matrix[:, :, -1], key_layer)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = jax.nn.softmax(last_product, axis=-1)  # [bsz, n_heads, from_block_size, n]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        last_context_layer = jnp.einsum("bhqk,bhkd->bhqd", last_attn_weights, value_layer)
        last_context_layer = jnp.expand_dims(last_context_layer, 2)

        # combining representations of all tokens
        context_layer = jnp.concatenate(
            [first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer],
            axis=2,
        )
        context_layer = context_layer.reshape(bsz, n_heads, from_seq_len, -1) * from_mask
        context_layer = jnp.transpose(context_layer, axes=(0, 2, 1, 3)).reshape(bsz, from_seq_len, -1)

        attention_probs = None

        return context_layer, attention_probs

    @staticmethod
    def jax_gather(params, indices, batch_dims=2):
        """
        Gather the indices from params correctly (equivalent to tf.gather but with modifications)

        Args:
            params: (bsz, n_heads, num_blocks, block_size, head_dim)
            indices: (<num_blocks, 1)
        """

        def _jax_gather(params, indices):
            return params[indices]

        for _ in range(batch_dims):
            _jax_gather = jax.vmap(_jax_gather, in_axes=(0, 0))

        return _jax_gather(params, indices)  # params.shape[:batch_dims] + indices.shape + params.shape[batch_dims+1:]

    def _create_rand_mask_from_inputs(
        self,
        from_blocked_mask,
        to_blocked_mask,
        broadcasted_rand_attn,
        num_attention_heads,
        num_random_blocks,
        batch_size,
        from_seq_length,
        from_block_size,
    ):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_blocked_mask: 2D Tensor of shape [batch_size, from_seq_length//from_block_size, from_block_size].
            to_blocked_mask: int32 Tensor of shape [batch_size, to_seq_length//to_block_size, to_block_size].
            broadcasted_rand_attn: [batch_size, num_attention_heads, from_seq_length//from_block_size-2, num_rand_blocks]
            num_attention_heads: int. Number of attention heads.
            num_random_blocks: int. Number of random chunks per row.
            batch_size: int. Batch size for computation.
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.

        Returns:
            float Tensor of shape [batch_size, num_attention_heads, from_seq_length//from_block_size-2,
            from_block_size, num_rand_blocks*to_block_size].
        """
        num_windows = from_seq_length // from_block_size - 2
        rand_mask = self.jax_gather(to_blocked_mask, broadcasted_rand_attn, batch_dims=1)
        rand_mask = rand_mask.reshape(
            batch_size, num_attention_heads, num_windows, num_random_blocks * from_block_size
        )
        rand_mask = jnp.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        """
        Gives the plan of where to put random attention.

        Args:
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.
            num_rand_blocks: int. Number of random chunks per row.

        Returns:
            plan_from_length: ending location of from block plan_num_rand_blocks: number of random ending location for
            each block
        """

        plan_from_length = []
        plan_num_rand_blocks = []
        if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)

        return plan_from_length, plan_num_rand_blocks

    @staticmethod
    def _bigbird_block_rand_mask(
        from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
    ):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_rand_blocks: int. Number of random chunks per row.
            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks chosen only up to last_idx.

        Returns:
            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
        """
        # using this method when from_seq_length in [1024, 3072, 4096]

        assert (
            from_seq_length // from_block_size == to_seq_length // to_block_size
        ), "Error the number of blocks needs to be same!"

        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # shorthand
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                elif (end + 1) == last:
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                else:
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                    )[:r]
        return rand_attn

    def _bigbird_block_rand_mask_with_head(
        self,
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        num_heads,
        plan_from_length,
        plan_num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_top=1,
        global_block_bottom=1,
        global_block_left=1,
        global_block_right=1,
    ):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_heads: int. total number of heads.
            plan_from_length: list. plan from length where num_random_blocks are choosen from.
            plan_num_rand_blocks: list. number of rand blocks within the plan.
            window_block_left: int. number of blocks of window to left of a block.
            window_block_right: int. number of blocks of window to right of a block.
            global_block_top: int. number of blocks at the top.
            global_block_bottom: int. number of blocks at the bottom.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.

        Returns:
            adjacency list of size num_head where each element is of size from_seq_length//from_block_size-2 by
            num_rand_blocks
        """
        # using this method when from_seq_length not in [1024, 3072, 4096]

        assert (
            from_seq_length // from_block_size == to_seq_length // to_block_size
        ), "Error the number of blocks needs to be same!"

        assert from_seq_length in plan_from_length, "Error from sequence length not in plan!"

        # Total number of blocks in the mmask
        num_blocks = from_seq_length // from_block_size
        # Number of blocks per plan
        plan_block_length = np.array(plan_from_length) // from_block_size
        # till when to follow plan
        max_plan_idx = plan_from_length.index(from_seq_length)
        # Random Attention adjacency list
        rand_attn = [
            np.zeros((num_blocks, np.sum(plan_num_rand_blocks[: max_plan_idx + 1])), dtype=np.int32)
            for i in range(num_heads)
        ]

        # We will go iteratively over the plan blocks and pick random number of
        # Attention blocks from the legally allowed blocks
        for plan_idx in range(max_plan_idx + 1):
            rnd_r_cnt = 0
            if plan_idx > 0:
                # set the row for all from_blocks starting from 0 to
                # plan_block_length[plan_idx-1]
                # column indx start fromm plan_block_length[plan_idx-1] and ends at
                # plan_block_length[plan_idx]
                if plan_num_rand_blocks[plan_idx] > 0:
                    rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
                    for blk_rw_idx in range(global_block_top, plan_block_length[plan_idx - 1]):
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=plan_block_length[plan_idx - 1],
                                to_end_block_id=plan_block_length[plan_idx],
                                num_rand_blocks=plan_num_rand_blocks[plan_idx],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )

                for pl_id in range(plan_idx):
                    if plan_num_rand_blocks[pl_id] == 0:
                        continue
                    for blk_rw_idx in range(plan_block_length[plan_idx - 1], plan_block_length[plan_idx]):
                        rnd_r_cnt = 0
                        to_start_block_id = 0
                        if pl_id > 0:
                            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                            to_start_block_id = plan_block_length[pl_id - 1]
                        curr_r_cnt = int(np.sum(plan_num_rand_blocks[: pl_id + 1]))
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=to_start_block_id,
                                to_end_block_id=plan_block_length[pl_id],
                                num_rand_blocks=plan_num_rand_blocks[pl_id],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )

            if plan_num_rand_blocks[plan_idx] == 0:
                continue
            curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
            from_start_block_id = global_block_top
            to_start_block_id = 0
            if plan_idx > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                from_start_block_id = plan_block_length[plan_idx - 1]
                to_start_block_id = plan_block_length[plan_idx - 1]

            for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
                for h in range(num_heads):
                    rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                        block_id=blk_rw_idx,
                        to_start_block_id=to_start_block_id,
                        to_end_block_id=plan_block_length[plan_idx],
                        num_rand_blocks=plan_num_rand_blocks[plan_idx],
                        window_block_left=window_block_left,
                        window_block_right=window_block_right,
                        global_block_left=global_block_left,
                        global_block_right=global_block_right,
                    )

        for nh in range(num_heads):
            rand_attn[nh] = rand_attn[nh][global_block_top : num_blocks - global_block_bottom, :]

        return rand_attn

    @staticmethod
    def _get_single_block_row_attention(
        block_id,
        to_start_block_id,
        to_end_block_id,
        num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_left=1,
        global_block_right=1,
    ):
        """
        For a single row block get random row attention.

        Args:
            block_id: int. block id of row.
            to_start_block_id: int. random attention column start id.
            to_end_block_id: int. random attention column end id.
            num_rand_blocks: int. number of random blocks to be selected.
            window_block_left: int. number of blocks of window to left of a block.
            window_block_right: int. number of blocks of window to right of a block.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.

        Returns:
            row containing the random attention vector of size num_rand_blocks.
        """
        # list of to_blocks from which to choose random attention
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        # permute the blocks
        perm_block = np.random.permutation(to_block_list)

        # illegal blocks for the current block id, using window
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

        # Add blocks at the start and at the end
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

        # The second from_block cannot choose random attention on second last to_block
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)

        # The second last from_block cannot choose random attention on second to_block
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)

        selected_random_blokcs = []

        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blokcs.append(perm_block[i])
            if len(selected_random_blokcs) == num_rand_blocks:
                break
        return np.array(selected_random_blokcs, dtype=np.int32)


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertSelfOutput with Bert->BigBird
class FlaxBigBirdSelfOutput(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FlaxBigBirdAttention(nn.Module):
    config: BigBirdConfig
    layer_id: int = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.config.attention_type == "original_full":
            self.self = FlaxBigBirdSelfAttention(self.config, dtype=self.dtype)
        elif self.config.attention_type == "block_sparse":
            self.self = FlaxBigBirdBlockSparseAttention(self.config, block_sparse_seed=self.layer_id, dtype=self.dtype)
        else:
            raise ValueError(
                f"Your `config.attention_type` is {self.config.attention_type} but it can either be `original_full` or `block_sparse`"
            )

        self.output = FlaxBigBirdSelfOutput(self.config, dtype=self.dtype)

    # Copied from transformers.models.bert.modeling_flax_bert.FlaxBertAttention.__call__ with Bert->BigBird
    def __call__(self, hidden_states, attention_mask=None, deterministic=True, output_attentions: bool = False):
        # Attention mask comes in as attention_mask.shape == (*batch_sizes, kv_length)
        # FLAX expects: attention_mask.shape == (*batch_sizes, 1, 1, kv_length) such that it is broadcastable
        # with attn_weights.shape == (*batch_sizes, num_heads, q_length, kv_length)
        attn_outputs = self.self(
            hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertIntermediate with Bert->BigBird
class FlaxBigBirdIntermediate(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertOutput with Bert->BigBird
class FlaxBigBirdOutput(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


class FlaxBigBirdLayer(nn.Module):
    config: BigBirdConfig
    layer_id: int = None
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = FlaxBigBirdAttention(self.config, layer_id=self.layer_id, dtype=self.dtype)
        self.intermediate = FlaxBigBirdIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxBigBirdOutput(self.config, dtype=self.dtype)

    # Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLayer.__call__ with Bert->BigBird
    def __call__(self, hidden_states, attention_mask, deterministic: bool = True, output_attentions: bool = False):
        attention_outputs = self.attention(
            hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions
        )
        attention_output = attention_outputs[0]

        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


class FlaxBigBirdLayerCollection(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxBigBirdLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    # Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection.__call__ with Bert->BigBird
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertEncoder with Bert->BigBird
class FlaxBigBirdEncoder(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxBigBirdLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertPredictionHeadTransform with Bert->BigBird
class FlaxBigBirdPredictionHeadTransform(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.LayerNorm(hidden_states)


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLMPredictionHead with Bert->BigBird
class FlaxBigBirdLMPredictionHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transform = FlaxBigBirdPredictionHeadTransform(self.config, dtype=self.dtype)
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.transform(hidden_states)

        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)

        hidden_states += self.bias
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertOnlyMLMHead with Bert->BigBird
class FlaxBigBirdOnlyMLMHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxBigBirdLMPredictionHead(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states


class FlaxBigBirdPreTrainingHeads(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxBigBirdLMPredictionHead(self.config, dtype=self.dtype)
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, hidden_states, pooled_output, shared_embedding=None):
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class FlaxBigBirdPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BigBirdConfig
    base_model_prefix = "bert"
    module_class: nn.Module = None

    def __init__(
        self,
        config: BigBirdConfig,
        input_shape: Optional[tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        if config.attention_type == "block_sparse" and input_shape is None:
            input_shape = (1, 12 * config.block_size)
        elif input_shape is None:
            input_shape = (1, 1)

        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        attention_mask = jnp.ones_like(input_ids)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, input_ids, attention_mask, token_type_ids, position_ids, return_dict=False)[
            "params"
        ]

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


class FlaxBigBirdModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxBigBirdEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBigBirdEncoder(self.config, dtype=self.dtype)
        self.pooler = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        pooled = nn.tanh(self.pooler(hidden_states[:, 0, :])) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    "The bare BigBird Model transformer outputting raw hidden-states without any specific head on top.",
    BIG_BIRD_START_DOCSTRING,
)
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertModel with Bert->BigBird
class FlaxBigBirdModel(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdModule


append_call_sample_docstring(
    FlaxBigBirdModel, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC
)


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingModule with Bert->BigBird
class FlaxBigBirdForPreTrainingModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBigBirdModule(config=self.config, dtype=self.dtype)
        self.cls = FlaxBigBirdPreTrainingHeads(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):

        # Model
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        hidden_states = outputs[0]
        pooled_output = outputs[1]

        prediction_scores, seq_relationship_score = self.cls(
            hidden_states, pooled_output, shared_embedding=shared_embedding
        )

        if not return_dict:
            return (prediction_scores, seq_relationship_score) + outputs[2:]

        return FlaxBigBirdForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    BigBird Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BIG_BIRD_START_DOCSTRING,
)
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForPreTraining with Bert->BigBird
class FlaxBigBirdForPreTraining(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForPreTrainingModule


FLAX_BIG_BIRD_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example::

        >>> from transformers import BigBirdTokenizer, FlaxBigBirdForPreTraining

        >>> tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
        >>> model = FlaxBigBirdForPreTraining.from_pretrained('google/bigbird-roberta-base')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
"""

overwrite_call_docstring(
    FlaxBigBirdForPreTraining,
    BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_BIG_BIRD_FOR_PRETRAINING_DOCSTRING,
)
append_replace_return_docstrings(
    FlaxBigBirdForPreTraining, output_type=FlaxBigBirdForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
)


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForMaskedLMModule with Bert->BigBird
class FlaxBigBirdForMaskedLMModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBigBirdModule(config=self.config, add_pooling_layer=False, dtype=self.dtype)
        self.cls = FlaxBigBirdOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("""BigBird Model with a `language modeling` head on top. """, BIG_BIRD_START_DOCSTRING)
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForMaskedLM with Bert->BigBird
class FlaxBigBirdForMaskedLM(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForMaskedLMModule


append_call_sample_docstring(
    FlaxBigBirdForMaskedLM, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC
)


class FlaxBigBirdClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.out_proj = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, features, deterministic=True):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x, deterministic=deterministic)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.out_proj(x)
        return x


class FlaxBigBirdForSequenceClassificationModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBigBirdModule(config=self.config, dtype=self.dtype)
        self.classifier = FlaxBigBirdClassificationHead(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, deterministic=deterministic)

        if not return_dict:
            return (logits,) + outputs[2:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    BigBird Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    BIG_BIRD_START_DOCSTRING,
)
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForSequenceClassification with Bert->BigBird
class FlaxBigBirdForSequenceClassification(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForSequenceClassificationModule


append_call_sample_docstring(
    FlaxBigBirdForSequenceClassification,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForMultipleChoiceModule with Bert->BigBird
class FlaxBigBirdForMultipleChoiceModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBigBirdModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(1, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # Model
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.reshape(-1, num_choices)

        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    BigBird Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForMultipleChoice(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForMultipleChoiceModule

    def __init__(
        self,
        config: BigBirdConfig,
        input_shape: Optional[tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs
    ):
        if config.attention_type == "block_sparse" and input_shape is None:
            input_shape = (1, 1, 12 * config.block_size)
        elif input_shape is None:
            input_shape = (1, 1)
        super().__init__(config, input_shape=input_shape, seed=seed, dtype=dtype)


overwrite_call_docstring(
    FlaxBigBirdForMultipleChoice, BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
append_call_sample_docstring(
    FlaxBigBirdForMultipleChoice,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForTokenClassificationModule with Bert->BigBird
class FlaxBigBirdForTokenClassificationModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBigBirdModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    BigBird Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BIG_BIRD_START_DOCSTRING,
)
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForTokenClassification with Bert->BigBird
class FlaxBigBirdForTokenClassification(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForTokenClassificationModule


append_call_sample_docstring(
    FlaxBigBirdForTokenClassification,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)


class FlaxBigBirdForQuestionAnsweringHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.intermediate = FlaxBigBirdIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxBigBirdOutput(self.config, dtype=self.dtype)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, encoder_output, deterministic=True):
        hidden_states = self.dropout(encoder_output, deterministic=deterministic)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states, encoder_output)
        hidden_states = self.qa_outputs(hidden_states)
        return hidden_states


class FlaxBigBirdForQuestionAnsweringModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    add_pooling_layer: bool = False

    def setup(self):
        self.config.num_labels = 2
        self.bert = FlaxBigBirdModule(self.config, dtype=self.dtype, add_pooling_layer=self.add_pooling_layer)
        self.qa_classifier = FlaxBigBirdForQuestionAnsweringHead(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        logits_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):

        # Model
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pooled_output = outputs[1] if self.add_pooling_layer else None
        logits = self.qa_classifier(hidden_states, deterministic=deterministic)

        if logits_mask is not None:
            # removing question tokens from the competition
            logits = logits - logits_mask * 1e6

        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        return FlaxBigBirdForQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            pooled_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    BigBird Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForQuestionAnswering(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForQuestionAnsweringModule

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        question_lengths=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        if question_lengths is None and input_ids is not None:
            # assuming input_ids format: <cls> <question> <sep> context <sep>
            question_lengths = jnp.argmax((input_ids == self.config.sep_token_id).astype("i4"), axis=-1) + 1
            question_lengths = jnp.expand_dims(question_lengths, axis=1)

        seqlen = input_ids.shape[1]

        logits_mask = None
        if question_lengths is not None:
            # setting lengths logits to `-inf`
            logits_mask = self.prepare_question_mask(question_lengths, seqlen)
            if token_type_ids is None:
                token_type_ids = (~logits_mask).astype("i4")
            logits_mask = jnp.expand_dims(logits_mask, axis=2)

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            token_type_ids,
            jnp.array(position_ids, dtype="i4"),
            logits_mask,
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )

    @staticmethod
    def prepare_question_mask(q_lengths, maxlen: int):
        # q_lengths -> (bz, 1)
        mask = jnp.arange(0, maxlen)
        mask = jnp.expand_dims(mask, axis=0) < q_lengths
        return mask


append_call_sample_docstring(
    FlaxBigBirdForQuestionAnswering,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxBigBirdForQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)
