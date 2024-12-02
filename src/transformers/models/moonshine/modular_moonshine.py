from ...configuration_utils import PretrainedConfig
from ..phi.modeling_phi import PhiAttention, PhiFlashAttention2, PhiSdpaAttention, PhiMLP, PhiRotaryEmbedding
from ..llama.modeling_llama import LlamaDecoderLayer
from ..mistral.modeling_mistral import MistralMLP
from ..whisper.modeling_whisper import WhisperEncoder

from ...cache_utils import Cache, DynamicCache, StaticCache

from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel

import torch.nn as nn
import torch

from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from typing import Optional, Tuple

from ...activations import ACT2FN

import copy
class MoonshineConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MoonshineModel`]. It is used to instantiate a Moonshine
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Moonshine
    [UsefulSensors/moonshine](https://huggingface.co/UsefulSensors/moonshine).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the Moonshine model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MoonshineModel`].
        hidden_size (`int`, *optional*, defaults to 288):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder and decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder and decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        encoder_hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder.
        decoder_hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. TODO: check this
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings. TODO: check this
        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
            Percentage of the query and keys which will have rotary embedding. TODO: check this
        ff_mult (`int`, *optional*, defaults to 4):
            Factor by which to scale the intermediate size.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        qk_layernorm (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize the Queries and Keys after projecting the hidden states.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        bos_token_id (`int`, *optional*, defaults to 1):
            Denotes beginning of sequences token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            Denotes end of sequences token id.

    Example:

    ```python
    >>> from transformers import MoonshineModel, MoonshineConfig

    >>> # Initializing a Moonshine style configuration
    >>> configuration = MoonshineConfig().from_pretrained("UsefulSensors/moonshine")

    >>> # Initializing a model from the configuration
    >>> model = MoonshineModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moonshine"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=288,
        intermediate_size=None,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=None,
        encoder_hidden_act="gelu",
        decoder_hidden_act="silu",
        max_position_embeddings=2048,
        layer_norm_eps=1e-5,
        use_cache=True,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        attention_bias=False,
        attention_dropout=0.0,
        qk_layernorm=False,
        rope_scaling=None,
        ff_mult=4,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.encoder_hidden_act = encoder_hidden_act
        self.decoder_hidden_act = decoder_hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.qk_layernorm = qk_layernorm
        self.rope_scaling = rope_scaling
        self.ff_mult = ff_mult

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_every_two(q) * sin)
    k_embed = (k * cos) + (rotate_every_two(k) * sin)
    return q_embed, k_embed


class MoonshineRotaryEmbedding(PhiRotaryEmbedding):
    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.stack((freqs, freqs), dim=-1)
            emb = emb.flatten(-2) # in einsum notation: rearrange(x, '... d j -> ... (d j)')
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MoonshineNonGatedMLP(PhiMLP):
    def __init__(self, config: MoonshineConfig, hidden_act: str):
        config = copy.deepcopy(config)
        config.hidden_act = hidden_act
        if config.intermediate_size is None:
            config.intermediate_size = config.hidden_size * config.ff_mult
        super().__init__(config)


class MoonshineGatedMLP(MistralMLP):
    def __init__(self, config: MoonshineConfig, hidden_act: str):
        config = copy.deepcopy(config)
        config.hidden_act = hidden_act
        if config.intermediate_size is None:
            config.intermediate_size = config.hidden_size * config.ff_mult * 2
        super().__init__(config)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
    

class MoonshineMLP:
    def __new__(cls, config: MoonshineConfig, hidden_act: str):
        if hidden_act == "gelu":  
            return MoonshineNonGatedMLP(config, hidden_act)
        elif hidden_act == "silu":
            return MoonshineGatedMLP(config, hidden_act)
        else:
            raise ValueError(f"Unsupported activation function: {hidden_act}, please use 'gelu' or 'silu'")


class MoonshineAttention(PhiAttention):
    def __init__(self, config: MoonshineConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_ndims = max(config.hidden_size // config.num_attention_heads // 2, 32)

        self.rotary_emb = MoonshineRotaryEmbedding(
            dim=self.rotary_ndims, 
            max_position_embeddings=config.max_position_embeddings,
        )


class MoonshineFlashAttention2(PhiFlashAttention2):
    pass


class MoonshineSdpaAttention(PhiSdpaAttention):
    pass


MOONSHINE_ATTENTION_CLASSES = {
    "eager": MoonshineAttention,
    "flash_attention_2": MoonshineFlashAttention2,
    "sdpa": MoonshineSdpaAttention,
}


class MoonshineEncoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MoonshineConfig, layer_idx: int):
        super().__init__(config, layer_idx)
       
        self.mlp = MoonshineMLP(config, config.encoder_hidden_act)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)


class MoonshineDecoderLayer(nn.Module):
    def __init__(self, config: MoonshineConfig, layer_idx: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MOONSHINE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.encoder_attn = MOONSHINE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = MoonshineMLP(config, config.decoder_hidden_act) 
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = residual + hidden_states

            # add cross-attn to positions 1 of present_key_value tuple
            present_key_value = (present_key_value, cross_attn_present_key_value)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MoonshineEncoder(PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`MoonshineEncoderLayer`].

    Args:
        config: MoonshineConfig
    """

    def __init__(self, config: MoonshineConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.conv1 = nn.Conv1d(1, embed_dim, kernel_size=127, stride=64, bias=False)
        self.conv2 = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=7, stride=3)
        self.conv3 = nn.Conv1d(2 * embed_dim, embed_dim, kernel_size=3, stride=2)
        self.groupnorm = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=1e-5)

        self.rotary_emb = MoonshineRotaryEmbedding(
            dim=max(config.hidden_size // config.num_attention_heads // 2, 32), 
            max_position_embeddings=config.max_position_embeddings, 
        ) 

        self.layers = nn.ModuleList([MoonshineEncoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps, bias=False)

        self.gradient_checkpointing = False
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, 1, sequence_length)`):
                Float values of the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`).
            attention_mask (`torch.Tensor`)`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1, 
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds = nn.functional.tanh(self.conv1(input_features))
        inputs_embeds = self.groupnorm(inputs_embeds)
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = nn.functional.gelu(self.conv3(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1) 
        hidden_states = inputs_embeds
        
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        embed_pos = self.rotary_emb(hidden_states, position_ids)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
       
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                    position_embeddings=embed_pos,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    None,
                    output_attentions=output_attentions,
                    position_embeddings=embed_pos,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )