"""
Granite classes form the basis for Megatron-based models used by IBM. The code is derived from GPTBigCode classes.
"""

import math
import warnings
from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from ....generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
)
from ....modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ....modeling_utils import PreTrainedModel
from ..configuration_granite import GraniteConfig
from ..enums import AttentionHeadType, AttentionImplementation, NormalizationImplementation, PositionEmbeddingType
from .attention import Attention
from .layer import GraniteBlock
from .mlp import MLP
from .normalization import RMSNorm, get_normalization_function
from .position_embedding import Alibi, RoPE, YaRNScaledRoPE
from .utils import check_list_type, flatten_and_convert_to_tensors


class GranitePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GraniteConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    causal = True
    _no_split_modules = ["GraniteBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, config: GraniteConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.attention_implementation = AttentionImplementation(kwargs.get("attention_implementation", "sdpa"))
        self.normalization_implementation = NormalizationImplementation(
            kwargs.get("normalization_implementation", "torch")
        )

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, GraniteModel):
            module.gradient_checkpointing = value

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, (MLP, Attention)):
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            module.c_proj.weight.data.normal_(
                mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer))
            )
            module.c_proj._is_hf_initialized = True
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def get_autoregressive_language_modeling_loss(
        self, lm_logits: torch.Tensor, labels: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        loss = None
        # Shift so that tokens < n predict n
        if labels is not None:
            if self.attention_implementation == AttentionImplementation.padding_free:
                shift_logits = lm_logits[:-1, :]
                shift_labels = labels[1:].to(shift_logits.device)

                # this is needed so that the last token of current example doesn't predict first token of next example
                drop_loss_positions = cu_seqlens[1:-1] - 1
                shift_labels[drop_loss_positions] = -100
            else:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def prepare_inputs_for_model(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]],
        inputs_embeds: Union[torch.Tensor, List[List[float]]],
        position_ids: Union[torch.Tensor, List[List[int]]],
        token_type_ids: Union[torch.Tensor, List[List[int]]],
        labels: Union[torch.Tensor, List[List[int]]],
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        attention_mask: torch.Tensor,
        head_mask: torch.Tensor,
        use_cache: bool,
        output_attentions: bool,
        attention_implementation: AttentionImplementation,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_implementation == AttentionImplementation.padding_free:
            if isinstance(input_ids, list) or isinstance(inputs_embeds, list):
                device = torch.cuda.current_device()

                # check input types are correct
                error_message = "{variable} should be of type List[List[{dtype}]]"
                check_list_type(input_ids, error_message.format(variable="input_ids", dtype="int"))
                check_list_type(inputs_embeds, error_message.format(variable="inputs_embeds", dtype="float"))
                check_list_type(position_ids, error_message.format(variable="position_ids", dtype="int"))
                check_list_type(token_type_ids, error_message.format(variable="token_type_ids", dtype="int"))
                check_list_type(labels, error_message.format(variable="labels", dtype="int"))

                # this is managed internally
                error_message = (
                    "{variable} should not be passed for flash attention when using List[List[int]] "
                    "input types attention mask logic is handled internally"
                )
                assert cu_seqlens is None, error_message.format(variable="cu_seqlens")
                assert max_seqlen is None, error_message.format(variable="max_seqlen")
                assert attention_mask is None, error_message.format(variable="attention_mask")

                # prepare inputs for the model
                seqlens = torch.tensor([0] + [len(x) for x in input_ids])
                cu_seqlens = seqlens.cumsum(dim=-1).to(device, torch.int32)
                max_seqlen = seqlens.max().to(device)

                if position_ids is None:
                    position_ids = [list(range(len(x))) for x in input_ids]
                position_ids = flatten_and_convert_to_tensors(position_ids, device)

                input_ids = flatten_and_convert_to_tensors(input_ids, device)

                if inputs_embeds is not None:
                    inputs_embeds = flatten_and_convert_to_tensors(inputs_embeds, device)

                if token_type_ids is not None:
                    token_type_ids = flatten_and_convert_to_tensors(token_type_ids, device)

                if labels is not None:
                    labels = flatten_and_convert_to_tensors(labels, device)
            else:
                assert (
                    cu_seqlens is not None
                ), "cu_seqlens needs to be specified when using tensor inputs with padding_free"
                assert position_ids is not None, "max_seqlen needs to be specified when specifying cu_seqlens"
                assert max_seqlen is not None, "max_seqlen needs to be specified when specifying cu_seqlens"
                assert attention_mask is None, "attention_mask should not be passed when specifying cu_seqlens"

            if use_cache or past_key_values is not None:
                raise NotImplementedError("KV caching is not supported with padding_free attention")

        error_message = "{variable} is only supported with math attention"

        if output_attentions:
            assert attention_implementation == AttentionImplementation.math, error_message.format(
                variable="output_attentions"
            )

        if head_mask is not None:
            assert attention_implementation == AttentionImplementation.math, error_message.format(variable="head_mask")

        return input_ids, position_ids, token_type_ids, labels, cu_seqlens, max_seqlen

    def extra_repr(self) -> str:
        return (
            f"attention_implementation = {self.attention_implementation.value}, "
            f"normalization_implementation = {self.normalization_implementation.value}"
        )

    def generate(
        self,
        inputs: torch.Tensor = None,
        generation_config: GenerationConfig = None,
        logits_processor: LogitsProcessorList = None,
        stopping_criteria: StoppingCriteriaList = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
        synced_gpus: bool = None,
        assistant_model: PreTrainedModel = None,
        streamer: BaseStreamer = None,
        negative_prompt_ids: torch.Tensor = None,
        negative_prompt_attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> Union[
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        GenerateBeamDecoderOnlyOutput,
        GenerateBeamEncoderDecoderOutput,
        torch.LongTensor,
    ]:
        assert (
            self.attention_implementation != AttentionImplementation.padding_free
        ), "padding_free attention implementation doesn't support generation"

        return super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            assistant_model,
            streamer,
            negative_prompt_ids,
            negative_prompt_attention_mask,
            **kwargs,
        )


class GraniteModel(GranitePreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config: GraniteConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        assert (
            self.embed_dim % self.num_heads == 0
        ), f"`embed_dim` ({self.embed_dim}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.embed_dim // self.num_heads

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                GraniteBlock(config, self.attention_implementation, self.normalization_implementation, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = get_normalization_function(
            config.normalization_function,
            self.embed_dim,
            eps=config.layer_norm_epsilon,
            normalization_implementation=self.normalization_implementation,
        )

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self._setup_positional_encoding()

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor = None,
        past_key_values: List[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        (
            output_attentions,
            output_hidden_states,
            use_cache,
            return_dict,
            input_shape,
            hidden_states,
            attention_mask,
            position_ids,
            head_mask,
            alibi_bias,
            rope_cos_sin,
            past_key_values,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # ==========================================================================================
        # AttentionImplementation.padding_free:
        #     attention_mask -> None
        # AttentionImplementation.flash:
        #     attention_mask -> (batch_size, key_length)
        # else:
        #     attention_mask -> (batch_size, 1, query_length, key_length)
        # ==========================================================================================

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = [] if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                outputs = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    position_ids,
                    alibi_bias,
                    rope_cos_sin,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                    cu_seqlens,
                    max_seqlen,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    alibi_bias=alibi_bias,
                    rope_cos_sin=rope_cos_sin,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

            hidden_states = outputs[0]
            if use_cache:
                presents.append(outputs[1])

            if output_attentions:
                all_self_attentions += (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _get_position_ids(
        self, attention_mask: torch.Tensor, past_length: int, query_length: int, key_length: int, device: torch.device
    ) -> torch.Tensor:
        if attention_mask is not None and len(attention_mask.shape) == 2:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_length > 0:
                position_ids = position_ids[:, past_length:key_length:]
        else:
            position_ids = torch.arange(past_length, key_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, query_length)

        return position_ids

    def _get_alibi_bias(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.position_embedding_type != PositionEmbeddingType.alibi:
            return None

        alibi_bias = self.alibi(attention_mask, batch_size, key_length, device, dtype)

        # ==========================================================================================
        # alibi_bias -> (batch_size, num_heads, key_length)
        # ==========================================================================================

        alibi_bias = alibi_bias.unsqueeze(2)
        if query_length != 1:
            alibi_bias = alibi_bias.expand(-1, -1, query_length, -1)

        # ==========================================================================================
        # alibi_bias -> (batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        return alibi_bias

    def _get_rope_cos_sin(self, key_length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.position_embedding_type == PositionEmbeddingType.rope:
            return self.rope(key_length, dtype=dtype, device=device)

    def _prepare_causal_attention_mask(
        self, attention_mask: torch.Tensor, batch_size: int, query_length: int, key_length: int, device: torch.device
    ) -> torch.Tensor:
        past_length = key_length - query_length

        # ==========================================================================================
        # attention_mask -> (batch_size, key_length)
        # ==========================================================================================

        if query_length > 1:
            # (query_length, key_length)
            causal_mask = torch.empty((query_length, key_length), dtype=torch.bool, device=device)
            causal_mask[:, past_length:] = torch.tril(
                torch.ones(query_length, query_length, dtype=torch.bool, device=device)
            )

            if past_length > 0:
                causal_mask[:, :past_length] = True

            # (query_length, key_length) -> (1, query_length, key_length)
            causal_mask = causal_mask.unsqueeze(0)

            if attention_mask is None:
                # (1, query_length, key_length) -> (batch_size, query_length, key_length)
                causal_mask = causal_mask.expand(batch_size, -1, -1)
            else:
                # (1, query_length, key_length) & (batch_size, 1, key_length) -> (batch_size, query_length, key_length)
                causal_mask = causal_mask & attention_mask.unsqueeze(1).to(torch.bool)
        else:
            if attention_mask is None:
                # (batch_size, query_length, key_length)
                causal_mask = torch.ones(batch_size, query_length, key_length, dtype=torch.bool, device=device)
            else:
                # (batch_size, query_length, key_length)
                causal_mask = attention_mask.unsqueeze(1).to(dtype=torch.bool, device=device)

        # ==========================================================================================
        # attention_mask -> (batch_size, query_length, key_length)
        # ==========================================================================================

        causal_mask = causal_mask.unsqueeze(1)

        # ==========================================================================================
        # attention_mask -> (batch_size, 1, query_length, key_length)
        # ==========================================================================================

        return causal_mask

    def _get_initial_hidden_state(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            inputs_embeds = inputs_embeds + self.wpe(position_ids)

        if token_type_ids is not None:
            inputs_embeds = inputs_embeds + self.wte(token_type_ids)

        inputs_embeds = self.drop(inputs_embeds)

        return inputs_embeds

    def _get_past_length(self, past_key_values: List[Union[torch.Tensor, Tuple[torch.Tensor]]]) -> int:
        if past_key_values is None:
            return 0

        past_length = past_key_values[0][0].size(2)
        return past_length

    def _prepare_a_bunch_of_stuff(
        self,
        input_ids: torch.Tensor = None,
        past_key_values: List[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> Tuple[
        bool,
        bool,
        bool,
        bool,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Union[Tuple[torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if use_cache is None:
            use_cache = (
                False
                if self.attention_implementation == AttentionImplementation.padding_free
                else self.config.use_cache
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()

            # special handling for packed flash attention with list inputs
            if self.attention_implementation == AttentionImplementation.padding_free:
                # for flash attention, there is no padding and we do packing
                # so, input_ids is of shape (s1 + s2 + ... + sb)
                batch_size = cu_seqlens.shape[0] - 1
            else:
                batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            # TODO special handling for packed flash attention needed here if we support inputs_embeds argument
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if self.attention_implementation == AttentionImplementation.padding_free:
            assert position_ids is not None, (
                "GraniteModel needs position_ids from outside when using flash attention with List[List[int]] "
                "inputs"
            )
        else:
            if self.position_embedding_type == PositionEmbeddingType.alibi:
                if position_ids is not None:
                    warnings.warn("`position_ids` have no functionality with Alibi.", FutureWarning)

            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, input_shape[-1])

        # ==========================================================================================
        # AttentionImplementation.padding_free:
        #     input_ids -> (total_q)
        #     attention_mask -> None
        #     position_ids -> (total_q)
        # else:
        #     input_ids -> (batch_size, query_length)
        #     attention_mask -> None or (batch_size, key_length)
        #     position_ids -> None or (batch_size, key_length)
        # ==========================================================================================

        past_length = None
        query_length = None
        key_length = None
        if self.attention_implementation == AttentionImplementation.padding_free:
            past_key_values = tuple([None] * len(self.h))
            key_length = max_seqlen
        else:
            past_length = self._get_past_length(past_key_values)
            if past_key_values is None:
                past_key_values = tuple([None] * len(self.h))

            query_length = input_shape[-1]
            key_length = past_length + query_length

        if position_ids is None:
            position_ids = self._get_position_ids(attention_mask, past_length, query_length, key_length, device)

        # ==========================================================================================
        # AttentionImplementation.padding_free:
        #     input_ids -> (total_q)
        #     attention_mask -> None
        #     position_ids -> (total_q)
        # else:
        #     input_ids -> (batch_size, query_length)
        #     attention_mask -> None or (batch_size, key_length)
        #     position_ids -> (batch_size, query_length)
        # ==========================================================================================

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        hidden_states = self._get_initial_hidden_state(input_ids, inputs_embeds, position_ids, token_type_ids)

        # ==========================================================================================
        # AttentionImplementation.padding_free:
        #     hidden_states -> (total_q, num_heads * head_dim)
        # else:
        #     hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        alibi_bias = self._get_alibi_bias(
            attention_mask, batch_size, query_length, key_length, device, hidden_states.dtype
        )

        # ==========================================================================================
        # alibi_bias -> (batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        rope_cos_sin = self._get_rope_cos_sin(key_length, dtype=hidden_states.dtype, device=hidden_states.device)

        # ==========================================================================================
        # AttentionImplementation.padding_free:
        #     rope_cos_sin -> 2 * (max_seqlen, head_dim)
        # else:
        #     rope_cos_sin -> 2 * (key_length, head_dim)
        # ==========================================================================================

        if self.attention_implementation == AttentionImplementation.flash:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
        elif self.attention_implementation == AttentionImplementation.padding_free:
            # no attention_mask is needed for list inputs
            attention_mask = None
        else:
            # for tensor inputs, we need the original attention_mask (and not the causal mask)

            # prepare causal mask only if not using flash attention
            attention_mask = self._prepare_causal_attention_mask(
                attention_mask, batch_size, query_length, key_length, device
            )

            if self.attention_implementation == AttentionImplementation.sdpa:
                attention_mask = torch.where(
                    attention_mask,
                    ~attention_mask if alibi_bias is None else alibi_bias,
                    self.h[0].attn._get_mask_value(attention_mask.device, hidden_states.dtype),
                )

        return (
            output_attentions,
            output_hidden_states,
            use_cache,
            return_dict,
            input_shape,
            hidden_states,
            attention_mask,
            position_ids,
            head_mask,
            alibi_bias,
            rope_cos_sin,
            past_key_values,
        )

    def _setup_positional_encoding(self) -> None:
        max_position_embeddings = self.config.max_position_embeddings

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            self.wpe = nn.Embedding(max_position_embeddings, self.embed_dim)
        elif self.position_embedding_type == PositionEmbeddingType.alibi:
            self.alibi = Alibi(self.num_heads)
        elif self.position_embedding_type == PositionEmbeddingType.rope:
            if self.config.rope_scaling is None:
                self.rope = RoPE(
                    self.head_dim, max_position_embeddings=max_position_embeddings, base=self.config.rope_theta
                )
            else:
                self.rope = YaRNScaledRoPE(
                    self.head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=self.config.rope_theta,
                    scale=self.config.rope_scaling["factor"],
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"],
                )
        else:
            raise NotImplementedError()


class GraniteForCausalLM(GranitePreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config: GraniteConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.transformer = GraniteModel(config, **kwargs)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.transformer.wte = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    # FIXME typing
    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, past_key_values=None, inputs_embeds: torch.Tensor = None, **kwargs
    ) -> dict:
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]] = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: Union[torch.Tensor, List[List[int]]] = None,
        position_ids: Union[torch.Tensor, List[List[int]]] = None,
        head_mask: torch.Tensor = None,
        inputs_embeds: Union[torch.Tensor, List[List[float]]] = None,
        labels: Union[torch.Tensor, List[List[int]]] = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, position_ids, token_type_ids, labels, cu_seqlens, max_seqlen = self.prepare_inputs_for_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            labels=labels,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            attention_implementation=self.attention_implementation,
        )

        # ==========================================================================================
        # AttentionImplementation.padding_free:
        #     input_ids -> (total_q)
        #     attention_mask -> None
        #     position_ids -> (total_q)
        # else:
        #     input_ids -> (batch_size, query_length)
        #     attention_mask -> None or (batch_size, key_length)
        #     position_ids -> None or (batch_size, key_length)
        # ==========================================================================================

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        loss = self.get_autoregressive_language_modeling_loss(lm_logits, labels, cu_seqlens)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(layer_past.index_select(0, beam_idx.to(layer_past.device)) for layer_past in past_key_values)
