from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs
from ..dinov2.modeling_dinov2 import Dinov2DropPath, Dinov2LayerScale
from ..dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTEmbeddings,
    DINOv3ViTRopePositionEmbedding,
    apply_rotary_pos_emb,
)
from ..vit.configuration_vit import ViTConfig
from ..vit.modeling_vit import (
    ViTAttention,
    ViTEncoder,
    ViTForImageClassification,
    ViTIntermediate,
    ViTLayer,
    ViTModel,
    ViTOutput,
    ViTPatchEmbeddings,
    ViTPreTrainedModel,
    ViTSelfAttention,
)


class ViTNepaConfig(ViTConfig):
    pass


@dataclass
class BaseModelOutputWithEmbedding(ModelOutput):
    """
    Base class for model outputs that include the last hidden states and input embeddings.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        input_embedding (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Input embeddings corresponding to the input tokens, before passing through the encoder layers.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    input_embedding: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class EmbeddedModelingOutput(ModelOutput):
    """
    Base class for outputs of embedding prediction.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Reconstruction loss.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class ViTNepaRopePositionEmbedding(DINOv3ViTRopePositionEmbedding):
    pass


class ViTNepaEmbeddings(DINOv3ViTEmbeddings):
    def __init__(self, config: ViTNepaConfig, use_mask_token: bool = False):
        super().__init__(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTNepaPatchEmbeddings(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=False)
        embeddings_clean = embeddings

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings_clean = torch.cat((cls_tokens, embeddings_clean), dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings, embeddings_clean


class ViTNepaPatchEmbeddings(ViTPatchEmbeddings):
    pass


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    output_attentions: bool = False,
    is_causal: bool = False,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # causal mask
    if is_causal:
        q_len, k_len = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.full((q_len, k_len), fill_value=float("-inf"), device=attn_weights.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_weights = attn_weights + causal_mask

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    outputs = (attn_output, attn_weights) if output_attentions else (attn_output, None)
    return outputs


class ViTNepaSelfAttention(ViTSelfAttention):
    def __init__(self, config: ViTNepaConfig):
        super().__init__(config)
        self.is_causal = config.is_causal
        self.q_norm = nn.LayerNorm(
            self.attention_head_size,
            eps=config.layer_norm_eps,
            elementwise_affine=config.qk_norm_affine,
            bias=config.qk_norm_bias,
        )
        self.k_norm = nn.LayerNorm(
            self.attention_head_size,
            eps=config.layer_norm_eps,
            elementwise_affine=config.qk_norm_affine,
            bias=config.qk_norm_bias,
        )

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

        query_layer = self.q_norm(query_layer)
        key_layer = self.k_norm(key_layer)

        # Apply RoPE to query and key
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            None,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs


class ViTNepaAttention(ViTAttention):
    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states, position_embeddings)
        output = self.output(self_attn_output, hidden_states)
        return output


class ViTNepaLayerScale(Dinov2LayerScale):
    pass


class ViTNepaIntermediate(ViTIntermediate):
    def __init__(self, config: ViTNepaConfig):
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.use_gated_mlp = config.use_gated_mlp

        if self.use_gated_mlp:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)

        super().__init__(config)
        del self.dense

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_out = self.up_proj(hidden_states)
        if self.use_gated_mlp:
            gate = self.gate_proj(hidden_states)
            gate_out = self.intermediate_act_fn(gate)
            hidden_states = gate_out * up_out
        else:
            hidden_states = self.intermediate_act_fn(up_out)
        return hidden_states


class ViTNepaDropPath(Dinov2DropPath):
    pass


class ViTNepaOutput(ViTOutput):
    def __init__(self, config: ViTNepaConfig, drop_path_rate: float = 0.0):
        super().__init__(config)
        self.layer_scale = ViTNepaLayerScale(config)
        self.drop_path = ViTNepaDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_scale(hidden_states)
        hidden_states = input_tensor + self.drop_path(hidden_states)
        return hidden_states


class ViTNepaLayer(ViTLayer):
    def __init__(self, config: ViTNepaConfig, drop_path_rate: float = 0.0):
        super().__init__(config)
        drop_path_rate = config.drop_path_rate
        self.drop_path = ViTNepaDropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.layer_scale = ViTNepaLayerScale(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm)
        attention_output = self.layer_scale(attention_output)
        attention_output = self.drop_path(attention_output)

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViTNepa, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        return layer_output


class ViTNepaEncoder(ViTEncoder):
    def __init__(self, config: ViTNepaConfig):
        super().__init__(config)
        drop_path_rates = [
            x.item() for x in torch.linspace(0, config.drop_path_prob, config.num_hidden_layers, device="cpu")
        ]
        self.layer = nn.ModuleList(
            [ViTNepaLayer(config, drop_path_rate=drop_path_rate) for drop_path_rate in range(drop_path_rates)]
        )

    def forward(self, hidden_states: torch.Tensor, positional_embeddings: torch.Tensor) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, positional_embeddings)

        return BaseModelOutput(last_hidden_state=hidden_states)


class ViTNepaPreTrainedModel(ViTPreTrainedModel):
    base_model_prefix = "vit_nepa"


class ViTNepaModel(ViTModel):
    def __init__(self, config: ViTNepaConfig):
        super().__init__(config)
        self.rope_embeddings = ViTNepaRopePositionEmbedding(config)
        del self.pooler

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithEmbedding:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_input, embedding_clean = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )
        position_embeds = self.rope_embeddings(pixel_values)

        encoder_outputs: BaseModelOutput = self.encoder(embedding_input, position_embeds)
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        return BaseModelOutputWithEmbedding(last_hidden_state=sequence_output, input_embedding=embedding_clean)


def prediction_loss(h_in, h_out, shift: bool = True):
    """
    similarity loss between two hidden states.

    Args:
        h_in:  [B, T, D]  input hidden states
        h_out: [B, T, D]  output hidden states (prediction)
        shift: if True, compare h_out[:, :-1] with h_in[:, 1:]
               else, compare h_out with h_in (position-wise)

    Returns:
        scalar loss (negative cosine similarity)
    """
    # detach target
    h_in = h_in.detach()

    if shift:
        # shift one step forward
        p = h_out[:, :-1, :]  # predict next
        z = h_in[:, 1:, :]  # target is next hidden state
    else:
        # same-position matching
        p = h_out
        z = h_in

    # normalize
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    # negative cosine similarity
    loss = -(p * z).sum(dim=-1).mean()
    return loss


class ViTNepaForPreTraining(ViTNepaPreTrainedModel):
    def __init__(self, config: ViTNepaConfig):
        super().__init__(config)

        self.vit_nepa = ViTNepaModel(config)
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> EmbeddedModelingOutput:
        r"""
        TODO
        ```"""

        outputs: BaseModelOutputWithEmbedding = self.vit_nepa(
            pixel_values,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            interpolate_pos_encoding=interpolate_pos_encoding,
            is_pretraining=True,
            **kwargs,
        )

        sequence_input = outputs.input_embedding
        sequence_output = outputs.last_hidden_state

        embedded_loss = prediction_loss(sequence_input, sequence_output)

        return EmbeddedModelingOutput(
            loss=embedded_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ViTNepaForImageClassification(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.add_pooling_layer = config.add_pooling_layer
        self.num_image_tokens = (config.image_size // config.patch_size) ** 2
        self.vit_nepa = ViTNepaModel(config)
        self.pooler = lambda hidden_states: hidden_states.mean(dim=1) if config.add_pooling_layer else None
        self.fc_norm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.add_pooling_layer else None
        )
        del self.vi_t_nepa

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        outputs: BaseModelOutputWithEmbedding = self.vit_nepa(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        if self.add_pooling_layer:
            image_tokens = sequence_output[:, -self.num_image_tokens :, :]
            pooled_output = image_tokens.mean(dim=1)
            pooled_output = self.fc_norm(pooled_output)
        else:
            pooled_output = sequence_output[:, -1, :]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "ViTNepaForImageClassification",
    "ViTNepaForPreTraining",
    "ViTNepaModel",
    "ViTNepaPreTrainedModel",
    "ViTNepaConfig",
]
