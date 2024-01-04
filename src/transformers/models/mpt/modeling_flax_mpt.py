import logging
import math
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxQuestionAnsweringModelOutput,
    FlaxTokenClassifierOutput,
)
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.models.mpt.configuration_mpt import MptConfig


logger = logging.getLogger(__name__)


class AttentionMaskConverter:
    def __init__(self, is_causal, sliding_window=None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

    def to_4d(self, mask_2d, query_length, key_value_length, dtype):
        mask_4d = jnp.expand_dims(mask_2d, axis=1)
        mask_4d = jnp.expand_dims(mask_4d, axis=2)
        mask_4d = jnp.broadcast_to(mask_4d, (mask_4d.shape[0], 1, query_length, key_value_length))
        mask_4d = mask_4d.astype(dtype)
        return mask_4d

    def to_causal_4d(self, batch_size, query_length, key_value_length, dtype):
        mask_4d = lax.convert_element_type(jnp.tril(jnp.ones((1, query_length, key_value_length), dtype=dtype)), dtype)
        if self.is_causal:
            mask_4d = jnp.tril(mask_4d, k=key_value_length - query_length)
        if self.sliding_window is not None:
            mask_4d = jnp.maximum(mask_4d, self.sliding_window_mask(query_length, key_value_length, dtype))
        return mask_4d


def _prepare_4d_causal_attention_mask(
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    sliding_window=None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`jnp.ndarray` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `jnp.ndarray`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`jnp.ndarray`):
            The embedded inputs as a JAX array.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    is_causal = True
    attn_mask_converter = AttentionMaskConverter(is_causal=is_causal, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if attention_mask.shape != expected_shape:
            raise ValueError(f"Incorrect 4D attention_mask shape: {attention_mask.shape}; expected: {expected_shape}.")
        else:
            # if the 4D mask has the correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = jnp.where(inverted_mask, -jnp.inf, attention_mask)
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype
        )

    return attention_mask


class FlaxMptAttention(nn.module):
    config: MptConfig
    attention_type: str
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.max_seq_length = config.max_seq_len
        self.head_dim = self.hidden_size // self.n_heads
        self.softmax_scale = config.attn_config.softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.n_heads)

        self.attn_dropout_p = config.attn_config.attn_pdrop

        self.Wqkv = nn.Dense(
            3 * self.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )
        self.out_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )

    def __call__(
        self,
        hidden_states,
        position_bias: jnp.ndarray,
        attention_mask: Optional[Tuple[jnp.ndarray]] = None,
        past_key_value: Optional[Tuple[jnp.ndarray]] = None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        mixed_qkv = self.Wqkv(hidden_states)
        query_states, key_states, value_states = jnp.split(mixed_qkv, 3, axis=-1)
        query_states = query_states.reshape((batch_size, seq_length, self.n_heads, self.head_dim)).transpose(
            (0, 2, 1, 3)
        )
        key_states = key_states.reshape((batch_size, seq_length, self.n_heads, self.head_dim)).transpose((0, 2, 1, 3))
        value_states = value_states.reshape((batch_size, seq_length, self.n_heads, self.head_dim)).transpose(
            (0, 2, 1, 3)
        )

        if past_key_value is not None:
            if past_key_value[0].shape[2] > 0:
                key_states = jnp.concatenate([past_key_value[0], key_states], axis=2)
                value_states = jnp.concatenate([past_key_value[1], value_states], axis=2)
            past_key_value = (key_states, value_states)
        else:
            past_key_value = (key_states, value_states)

        attention_scores = jnp.matmul(query_states, key_states.transpose((0, 1, 3, 2))) * self.softmax_scale

        query_length = seq_length if past_key_value is None else seq_length + past_key_value[0].shape[2]

        if position_bias is not None:
            if position_bias.ndim != 3:
                raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {position_bias.ndim}")
            key_length = key_states.shape[2]

            position_bias_query_index = max(0, position_bias.shape[1] - query_length)
            position_bias_key_index = max(0, position_bias.shape[2] - key_length)

            position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]

            attention_scores = attention_scores + position_bias

        if attention_mask is not None:
            attention_scores = jax.lax.select(
                attention_mask, jnp.full_like(attention_scores, -jnp.inf), attention_scores
            )

        attn_weights = nn.softmax(attention_scores, axis=-1).astype(value_states.dtype)
        attn_weights = nn.dropout(attn_weights, rate=self.attn_dropout_p, deterministic=not self.training)

        context_states = jnp.matmul(attn_weights, value_states)
        context_states = context_states.transpose((0, 2, 1, 3)).reshape((batch_size, seq_length, -1))
        attn_output = nn.Dense(features=self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(context_states)

        return attn_output, attn_weights, past_key_value


class FlaxMptMLP(nn.Module):
    hidden_size: int
    attn_pdrop: float

    def setup(self):
        self.up_proj = nn.Dense(
            features=4 * self.hidden_size, kernel_init=nn.initializers.xavier_uniform(), use_bias=False
        )
        self.act = nn.gelu
        self.down_proj = nn.Dense(
            features=self.hidden_size, kernel_init=nn.initializers.xavier_uniform(), use_bias=False
        )

    def forward(self, hidden_states, residual):
        hidden_states = self.act(self.up_proj(hidden_states))

        intermediate_output = self.down_proj(hidden_states)

        output = nn.dropout(intermediate_output, rate=self.attn_pdrop, deterministic=not self.training)
        output = output + residual

        return output


class FlaxMptBlock(nn.Module):
    hidden_size: int
    n_heads: int
    attn_pdrop: float
    layer_norm_epsilon: float

    def setup(self):
        self.norm_1 = nn.LayerNorm(epsilon=self.layer_norm_epsilon)
        self.attn = FlaxMptAttention(self.hidden_size, self.n_heads, self.attn_pdrop)
        self.norm_2 = nn.LayerNorm(epsilon=self.layer_norm_epsilon)
        self.ffn = FlaxMptMLP(self.hidden_size, self.attn_pdrop)
        self.resid_attn_dropout = nn.Dropout(rate=self.attn_pdrop)

    def forward(
        self,
        hidden_states,
        position_bias,
        attention_mask,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
    ):
        layernorm_output = self.norm_1(hidden_states)
        residual = hidden_states

        attn_outputs, attn_weights, past_key_value = self.attn(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            layer_past=layer_past,
        )

        hidden_states = self.resid_attn_dropout(attn_outputs) + residual

        layernorm_output = self.norm_2(hidden_states)
        residual = hidden_states

        output = self.ffn(layernorm_output, residual)
        outputs = (output,)

        if use_cache:
            outputs += (past_key_value,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class FlaxMptPreTrainedModel(FlaxPreTrainedModel):
    config_class = MptConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MptBlock"]
    _keys_to_ignore_on_load_missing = [r"lm_head.*."]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Dense):
            module.param = module.param.init(jax.random.PRNGKey(0), jnp.ones((1, module.param.shape[-1])))
            if module.bias is not None:
                module.bias = module.bias.init(jax.random.PRNGKey(0), jnp.zeros((1, module.bias.shape[-1])))
        elif isinstance(module, nn.Embed):
            module.weight = module.weight.init(jax.random.PRNGKey(0), jnp.ones((1, module.weight.shape[-1])))
            if module.padding_idx is not None:
                module.weight = module.weight.at[:, module.padding_idx].set(0.0)
        elif isinstance(module, nn.LayerNorm):
            module.param = module.param.init(jax.random.PRNGKey(0), jnp.ones((1, module.param.shape[-1])))
            if module.bias is not None:
                module.bias = module.bias.init(jax.random.PRNGKey(0), jnp.zeros((1, module.bias.shape[-1])))

    @staticmethod
    def _convert_to_mpt_cache(
        past_key_value: Tuple[Tuple[jnp.array, jnp.array]],
    ) -> Tuple[Tuple[jnp.array, jnp.array]]:
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        return tuple(
            (
                layer_past[0].reshape(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].reshape(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )


class FlaxMptModel(FlaxMptPreTrainedModel):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.n_heads

        self.wte = nn.Embed(self.config.vocab_size, self.config.hidden_size)

        self.blocks = [FlaxMptBlock(self.config, dtype=self.dtype) for _ in range(self.config.n_layers)]

        self.norm_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[Tuple[Tuple[jnp.ndarray, jnp.ndarray], ...]] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[jnp.ndarray, ...], FlaxBaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.blocks))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length_with_past), dtype=hidden_states.dtype)
        else:
            attention_mask = attention_mask.astype(hidden_states.dtype)

        alibi = self.build_mpt_alibi_tensor(self.num_heads, self.config.max_seq_len, device=hidden_states.device)

        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        causal_mask = causal_mask.astype(hidden_states.dtype)

        for block, layer_past in zip(self.blocks, past_key_values):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_bias=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class FlaxMptForCausalLMModule(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.transformer = FlaxMptModel(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxMptForCausalLM(FlaxMptPreTrainedModel):
    module_class = FlaxMptForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.Array] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)

        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since Mpt uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


class FlaxMptForTokenClassificationModule(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.transformer = FlaxMptModel(self.config, dtype=self.dtype)
        if hasattr(self.config, "classifier_dropout") and self.config.classifier_dropout is not None:
            classifier_dropout = self.config.classifier_dropout
        elif hasattr(self.config, "hidden_dropout") and self.config.hidden_dropout is not None:
            classifier_dropout = self.config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(rate=classifier_dropout, dtype=self.dtype)
        self.classifier = nn.Dense(
            self.config.num_labels,
            use_bias=True,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.xavier_uniform(),
        )

    def __call__(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **deprecated_arguments,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.astype(logits.dtype)
            batch_size, seq_length = labels.shape
            loss_fct = jax.nn.softmax_cross_entropy_with_logits
            loss = loss_fct(
                logits.reshape((batch_size * seq_length, self.config.num_labels)),
                labels.reshape((batch_size * seq_length,)),
            )

        if not return_dict:
            return (logits,) + outputs[2:]

        return FlaxTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxMptForTokenClassification(FlaxMptPreTrainedModel):
    module_class = FlaxMptForTokenClassificationModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.Array] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)

        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since Mpt uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


class FlaxMptForQuestionAnsweringModule(nn.Module):
    config: MptConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.transformer = FlaxMptModel(self.config, dtype=self.dtype)
        self.qa_outputs = nn.Dense(
            2,
            use_bias=True,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.xavier_uniform(),
        )

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = jnp.split(logits, indices_or_sections=2, axis=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if end_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # Sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = jnp.clip(start_positions, 0, ignored_index)
            end_positions = jnp.clip(end_positions, 0, ignored_index)

            loss_fct = jax.nn.softmax_cross_entropy_with_logits
            start_loss = jnp.mean(loss_fct(start_logits, start_positions))
            end_loss = jnp.mean(loss_fct(end_logits, end_positions))
            total_loss = (start_loss + end_loss) / 2.0

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return FlaxQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxMptForQuestionAnswering(FlaxMptPreTrainedModel):
    module_class = FlaxMptForQuestionAnsweringModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.Array] = None):
        # Initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)

        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since Mpt uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = jnp.cumsum(attention_mask, axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
