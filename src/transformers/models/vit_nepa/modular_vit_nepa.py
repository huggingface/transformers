from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from ... import PreTrainedConfig
from ... import initialization as init
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ..dinov2.modeling_dinov2 import Dinov2DropPath, Dinov2LayerScale
from ..dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTEmbeddings,
    DINOv3ViTRopePositionEmbedding,
    apply_rotary_pos_emb,
)
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


class ViTNepaConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTModel`]. It is used to instantiate an ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [SixAILab/nepa-base-patch14-224](https://huggingface.co/SixAILab/nepa-base-patch14-224) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        use_gated_mlp (`bool`, *optional*, defaults to `False`):
            Whether to use a gated MLP instead of a standard feed-forward block.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and
            `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        rope_theta (`float`, *optional*, defaults to 100.0):
            Base period used for rotary positional embeddings.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply normalization to the query and key projections before attention.
        qk_norm_bias (`bool`, *optional*, defaults to `False`):
            Whether the query/key normalization layers use a bias term.
        qk_norm_affine (`bool`, *optional*, defaults to `False`):
            Whether the query/key normalization layers use learnable affine parameters.
        layerscale_value (`float`, *optional*, defaults to 1e-05):
            Initial value for LayerScale factors. A non-positive value typically disables LayerScale.
        drop_path_prob (`float`, *optional*, defaults to 0.0):
            Stochastic depth (DropPath) rate used in the encoder blocks.
        add_pooling_layer (`bool`, *optional*, defaults to `False`):
            Whether to add a pooling layer on top of the final hidden state.
        is_causal (`bool`, *optional*, defaults to `True`):
            Whether to use a causal attention mask (for autoregressive-style training).
        pos_embed_shift (`float`, *optional*):
            Maximum magnitude of random positional embedding shift used as a training augmentation.
        pos_embed_jitter (`float`, *optional*):
            Amount of jitter applied to positional embedding coordinates as a training augmentation.
        pos_embed_rescale (`float`, *optional*, defaults to 2.0):
            Rescaling factor applied to positional embedding coordinates (e.g. when interpolating to new resolutions).

    Example:

    ```python
    >>> from transformers import ViTNepaConfig, ViTNepaModel

    >>> # Initializing a ViTNepa vit_nepa-base-patch16-224 style configuration
    >>> configuration = ViTNepaConfig()

    >>> # Initializing a model (with random weights) from the vit_nepa-base-patch16-224 style configuration
    >>> model = ViTNepaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vit_nepa"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        use_gated_mlp=False,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        rope_theta=100.0,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        qk_norm=False,
        qk_norm_bias=False,
        qk_norm_affine=False,
        layerscale_value=1e-5,
        drop_path_prob=0.0,
        add_pooling_layer=False,
        is_causal=True,
        pos_embed_shift: Optional[float] = None,
        pos_embed_jitter: Optional[float] = None,
        pos_embed_rescale: Optional[float] = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.use_gated_mlp = use_gated_mlp
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.qk_norm_bias = qk_norm_bias
        self.qk_norm_affine = qk_norm_affine
        self.layerscale_value = layerscale_value
        self.drop_path_prob = drop_path_prob
        self.add_pooling_layer = add_pooling_layer
        self.is_causal = is_causal
        self.pos_embed_shift = pos_embed_shift
        self.pos_embed_jitter = pos_embed_jitter
        self.pos_embed_rescale = pos_embed_rescale


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
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
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
    last_hidden_state: Optional[torch.FloatTensor] = None
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
        del self.register_tokens

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

    return attn_output, attn_weights


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
        self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
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
            **kwargs,
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
        self.drop_path = ViTNepaDropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.layer_scale = ViTNepaLayerScale(config)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm, position_embeddings)
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
        drop_path_rates = [
            x.item() for x in torch.linspace(0, config.drop_path_prob, config.num_hidden_layers, device="cpu")
        ]
        super().__init__(config)
        self.layer = nn.ModuleList(
            [ViTNepaLayer(config, drop_path_rate=drop_path_rate) for drop_path_rate in drop_path_rates]
        )

    def forward(self, hidden_states: torch.Tensor, positional_embeddings: torch.Tensor) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, positional_embeddings)

        return BaseModelOutput(last_hidden_state=hidden_states)


class ViTNepaPreTrainedModel(ViTPreTrainedModel):
    base_model_prefix = "vit_nepa"

    @torch.no_grad()
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, ViTNepaEmbeddings):
            init.trunc_normal_(module.cls_token, mean=0.0, std=self.config.initializer_range)
            if module.mask_token is not None:
                init.zeros_(module.mask_token)
        elif isinstance(module, ViTNepaRopePositionEmbedding):
            inv_freq = 1 / module.base ** torch.arange(0, 1, 4 / module.head_dim, dtype=torch.float32)
            init.copy_(module.inv_freq, inv_freq)


class ViTNepaModel(ViTModel):
    def __init__(self, config: ViTNepaConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config, add_pooling_layer, use_mask_token)
        self.rope_embeddings = ViTNepaRopePositionEmbedding(config)
        del self.pooler

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
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

        embedding_input, embedding_clean = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        position_embeds = self.rope_embeddings(pixel_values)

        encoder_outputs: BaseModelOutput = self.encoder(embedding_input, position_embeds)
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        return BaseModelOutputWithEmbedding(
            last_hidden_state=sequence_output,
            input_embedding=embedding_clean,
            attentions=encoder_outputs.attentions,
            hidden_states=encoder_outputs.hidden_states,
        )


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
        **kwargs: Unpack[TransformersKwargs],
    ) -> EmbeddedModelingOutput:
        outputs: BaseModelOutputWithEmbedding = self.vit_nepa(
            pixel_values,
            **kwargs,
        )

        sequence_input = outputs.input_embedding
        sequence_output = outputs.last_hidden_state

        loss = self.loss_function(sequence_input, sequence_output)

        return EmbeddedModelingOutput(
            loss=loss,
            last_hidden_state=outputs.last_hidden_state,
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        outputs: BaseModelOutputWithEmbedding = self.vit_nepa(
            pixel_values,
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
