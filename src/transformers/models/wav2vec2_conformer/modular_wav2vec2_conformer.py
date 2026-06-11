import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, Wav2Vec2BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, logging
from ...utils.output_capturing import OutputRecorder
from ..bert.modeling_bert import eager_attention_forward
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Adapter,
    Wav2Vec2AdapterLayer,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2FeedForward,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForCTC,
    Wav2Vec2ForPreTraining,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForXVector,
    Wav2Vec2GumbelVectorQuantizer,
    Wav2Vec2Model,
    Wav2Vec2PositionalConvEmbedding,
)
from .configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig


logger = logging.get_logger(__name__)

_HIDDEN_STATES_START_POSITION = 2


@auto_docstring(
    custom_intro="""
    Output type of [`Wav2Vec2ConformerForPreTraining`], with potential hidden states and attentions.
    """
)
@dataclass
class Wav2Vec2ConformerForPreTrainingOutput(ModelOutput):
    r"""
    loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
        Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
        paper](https://huggingface.co/papers/2006.11477).
    projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
        Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
        projected quantized states.
    projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
        Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
        target vectors for contrastive loss.
    codevector_perplexity (`torch.FloatTensor` of shape `(1,)`):
        The perplexity of the codevector distribution, used to measure the diversity of the codebook.
    contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
        The contrastive loss (L_m) as stated in the [official paper](https://huggingface.co/papers/2006.11477).
    diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
        The diversity loss (L_d) as stated in the [official paper](https://huggingface.co/papers/2006.11477).
    """

    loss: torch.FloatTensor | None = None
    projected_states: torch.FloatTensor | None = None
    projected_quantized_states: torch.FloatTensor | None = None
    codevector_perplexity: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    contrastive_loss: torch.FloatTensor | None = None
    diversity_loss: torch.FloatTensor | None = None


class Wav2Vec2ConformerPositionalConvEmbedding(Wav2Vec2PositionalConvEmbedding):
    pass


class Wav2Vec2ConformerRotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://huggingface.co/papers/2104.09864
    """

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        base = config.rotary_embedding_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def forward(self, hidden_states):
        sequence_length = hidden_states.shape[1]

        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        self.cached_sequence_length = sequence_length
        # Embeddings are computed in the dtype of the inv_freq constant
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)

        cos_embeddings = embeddings.cos()[:, None, None, :]
        sin_embeddings = embeddings.sin()[:, None, None, :]
        # Computed embeddings are cast to the dtype of the hidden state inputs
        self.cached_rotary_positional_embedding = torch.stack([cos_embeddings, sin_embeddings]).type_as(hidden_states)
        return self.cached_rotary_positional_embedding


class Wav2Vec2ConformerRelPositionalEmbedding(nn.Module):
    """Relative positional encoding module."""

    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_source_positions
        self.d_model = config.hidden_size
        self.register_buffer("pe", self.extend_pe(torch.tensor(0.0).expand(1, self.max_len)), persistent=False)

    def extend_pe(self, x, pe=None):
        # Reset the positional encodings
        if pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if pe.size(1) >= x.size(1) * 2 - 1:
                if pe.dtype != x.dtype or pe.device != x.device:
                    pe = pe.to(dtype=x.dtype, device=x.device)
                return pe
        # Suppose `i` is the position of query vector and `j` is the
        # position of key vector. We use positive relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.int64).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.int64).float() * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reverse the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://huggingface.co/papers/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        return pe.to(device=x.device, dtype=x.dtype)

    def forward(self, hidden_states: torch.Tensor):
        self.pe = self.extend_pe(hidden_states, self.pe)
        start_idx = self.pe.size(1) // 2 - hidden_states.size(1) + 1
        end_idx = self.pe.size(1) // 2 + hidden_states.size(1)
        relative_position_embeddings = self.pe[:, start_idx:end_idx]

        return relative_position_embeddings


class Wav2Vec2ConformerFeatureEncoder(Wav2Vec2FeatureEncoder):
    pass


class Wav2Vec2ConformerFeatureProjection(Wav2Vec2FeatureProjection):
    pass


class Wav2Vec2ConformerFeedForward(Wav2Vec2FeedForward):
    pass


class Wav2Vec2ConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=(config.conv_depthwise_kernel_size - 1) // 2,
            groups=config.hidden_size,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(config.conformer_conv_dropout)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


def _apply_relative_position_encoding(module, query, key, attention_mask, relative_position_embeddings):
    if relative_position_embeddings is None:
        raise ValueError(
            "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'relative'`"
        )

    # 1. project positional embeddings
    proj_relative_position_embeddings = module.linear_pos(relative_position_embeddings)
    proj_relative_position_embeddings = proj_relative_position_embeddings.view(
        relative_position_embeddings.size(0), -1, module.num_heads, module.head_size
    )
    proj_relative_position_embeddings = proj_relative_position_embeddings.permute(0, 2, 3, 1)

    # 2. compute matrix b and matrix d
    q_with_bias_v = query + module.pos_bias_v[None, :, None, :]
    relative_attention_scores = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)

    # 3. shift (skew) to get proper relative position indexing
    relative_attention_scores_shape = relative_attention_scores.shape
    relative_attention_scores = nn.functional.pad(relative_attention_scores, (1, 0)).view(
        *relative_attention_scores_shape[:2],
        relative_attention_scores_shape[3] + 1,
        relative_attention_scores_shape[2],
    )
    relative_attention_scores = relative_attention_scores[..., 1:, :].view(relative_attention_scores_shape)
    relative_attention_scores = relative_attention_scores[..., : key.size(2)]

    # 4. scale and combine with attention mask
    relative_attention_scores = relative_attention_scores * module.scaling
    if attention_mask is not None:
        relative_attention_scores = relative_attention_scores + attention_mask

    # 5. add pos_bias_u to query for the content-based attention (matrix a+c)
    query = query + module.pos_bias_u[None, :, None, :]

    return query, relative_attention_scores


class Wav2Vec2ConformerSelfAttention(nn.Module):
    """Construct an Wav2Vec2ConformerSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.head_size = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.position_embeddings_type = config.position_embeddings_type
        self.is_causal = False
        self.scaling = self.head_size**-0.5

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(p=config.attention_dropout)

        if self.position_embeddings_type == "relative":
            # linear transformation for positional encoding
            self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            # these two learnable bias are used in matrix c and matrix d
            # as described in https://huggingface.co/papers/1901.02860 Section 3.3
            self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
            self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        relative_position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # self-attention mechanism
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_size)

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
                )
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)

        query_states = self.linear_q(query_key_states).view(hidden_shape).transpose(1, 2)
        key_states = self.linear_k(query_key_states).view(hidden_shape).transpose(1, 2)
        value_states = self.linear_v(value_states).view(hidden_shape).transpose(1, 2)

        # apply relative position embeddings (matrix b+d) and bias the query (matrix a+c) if needed
        if self.position_embeddings_type == "relative":
            query_states, attention_mask = _apply_relative_position_encoding(
                self, query_states, key_states, attention_mask, relative_position_embeddings
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.linear_out(attn_output)

        return attn_output, attn_weights

    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        batch_size, sequence_length, hidden_size = hidden_states.size()
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)

        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]

        # rotate hidden_states with rotary embeddings
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.transpose(0, 1)

        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)

        return hidden_states


class Wav2Vec2ConformerEncoderLayer(GradientCheckpointingLayer):
    """Conformer block based on https://huggingface.co/papers/2005.08100."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = Wav2Vec2ConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = Wav2Vec2ConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module = Wav2Vec2ConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = Wav2Vec2ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask: torch.Tensor | None = None,
        relative_position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weigts = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            **kwargs,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states)
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, attn_weigts


class Wav2Vec2ConformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.position_embeddings_type == "relative":
            self.embed_positions = Wav2Vec2ConformerRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            self.embed_positions = Wav2Vec2ConformerRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None

        self.pos_conv_embed = Wav2Vec2ConformerPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Wav2Vec2ConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        **kwargs,
    ):
        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0.0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for layer in self.layers:
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and dropout_probability < self.config.layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    relative_position_embeddings=relative_position_embeddings,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class Wav2Vec2ConformerGumbelVectorQuantizer(Wav2Vec2GumbelVectorQuantizer):
    pass


class Wav2Vec2ConformerAdapter(Wav2Vec2Adapter):
    pass


class Wav2Vec2ConformerAdapterLayer(Wav2Vec2AdapterLayer):
    pass


@auto_docstring
class Wav2Vec2ConformerPreTrainedModel(PreTrainedModel):
    config: Wav2Vec2ConformerConfig
    base_model_prefix = "wav2vec2_conformer"
    main_input_name = "input_values"
    input_modalities = "audio"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": Wav2Vec2ConformerEncoderLayer,
        "attentions": OutputRecorder(Wav2Vec2ConformerSelfAttention, index=1, layer_name="encoder"),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        # Wav2Vec2ForPreTraining last 2 linear layers need standard Linear init.
        if isinstance(module, Wav2Vec2ConformerForPreTraining):
            module.project_hid.reset_parameters()
            module.project_q.reset_parameters()
        # gumbel softmax requires special init
        elif isinstance(module, Wav2Vec2ConformerGumbelVectorQuantizer):
            init.normal_(module.weight_proj.weight, mean=0.0, std=1)
            init.zeros_(module.weight_proj.bias)
            init.uniform_(module.codevectors)
        elif isinstance(module, Wav2Vec2ConformerSelfAttention):
            if hasattr(module, "pos_bias_u"):
                init.xavier_uniform_(module.pos_bias_u)
            if hasattr(module, "pos_bias_v"):
                init.xavier_uniform_(module.pos_bias_v)
        elif isinstance(module, Wav2Vec2ConformerPositionalConvEmbedding):
            init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            init.constant_(module.conv.bias, 0)
        elif isinstance(module, Wav2Vec2ConformerFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            init.uniform_(module.projection.weight, a=-k, b=k)
            init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
            init.zeros_(module.bias)
            init.ones_(module.weight)
            if getattr(module, "running_mean", None) is not None:
                init.zeros_(module.running_mean)
                init.ones_(module.running_var)
                init.zeros_(module.num_batches_tracked)
        elif isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, Wav2Vec2ConformerRotaryPositionalEmbedding):
            dim = self.config.hidden_size // self.config.num_attention_heads
            base = self.config.rotary_embedding_base
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
            init.copy_(module.inv_freq, inv_freq)
        elif isinstance(module, Wav2Vec2ConformerRelPositionalEmbedding):
            init.copy_(module.pe, module.extend_pe(torch.tensor(0.0).expand(1, module.max_len)))

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor | int, add_adapter: bool | None = None):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


WAV2VEC2_CONFORMER_START_DOCSTRING = None  # will be automatically redefined


Wav2Vec2ConformerBaseModelOutput = Wav2Vec2BaseModelOutput


class Wav2Vec2ConformerModel(Wav2Vec2ConformerPreTrainedModel, Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2ConformerConfig):
        Wav2Vec2ConformerPreTrainedModel.__init__(self, config)
        self.config = config
        self.feature_extractor = Wav2Vec2ConformerFeatureEncoder(config)
        self.feature_projection = Wav2Vec2ConformerFeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        self.encoder = Wav2Vec2ConformerEncoder(config)

        self.adapter = Wav2Vec2ConformerAdapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()


class Wav2Vec2ConformerForPreTraining(Wav2Vec2ForPreTraining):
    def __init__(self, config: Wav2Vec2ConformerConfig):
        super().__init__(config)


class Wav2Vec2ConformerForCTC(Wav2Vec2ForCTC):
    def __init__(self, config, target_lang: str | None = None):
        r"""
        target_lang (`str`, *optional*):
            Language id of adapter weights. Adapter weights are stored in the format adapter.<lang>.safetensors or
            adapter.<lang>.bin. Only relevant when using an instance of [`UniSpeechSatForCTC`] with adapters. Uses 'eng' by
            default.
        """
        super().__init__(config)

    def tie_weights(self):
        raise AttributeError("Not needed for Wav2Vec2Conformer")

    def freeze_base_model(self):
        raise AttributeError("Not needed for Wav2Vec2Conformer")


class Wav2Vec2ConformerForSequenceClassification(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)


class Wav2Vec2ConformerForAudioFrameClassification(Wav2Vec2ForAudioFrameClassification):
    def __init__(self, config):
        super().__init__(config)


class Wav2Vec2ConformerForXVector(Wav2Vec2ForXVector):
    def __init__(self, config):
        super().__init__(config)


__all__ = [
    "Wav2Vec2ConformerForAudioFrameClassification",
    "Wav2Vec2ConformerForCTC",
    "Wav2Vec2ConformerForPreTraining",
    "Wav2Vec2ConformerForSequenceClassification",
    "Wav2Vec2ConformerForXVector",
    "Wav2Vec2ConformerModel",
    "Wav2Vec2ConformerPreTrainedModel",
]
