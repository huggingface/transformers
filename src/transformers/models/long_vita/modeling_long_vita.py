# coding=utf-8

from typing import List, Optional, Tuple, Union
import torch

from ...cache_utils import Cache, DynamicCache
from .configuration_long_vita import LongVITAConfig
from .configuration_intern_vit import InternVisionConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from .modeling_intern_vit import InternVisionModel
from ...processing_utils import Unpack
from ..qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM
from .resampler_projector import ResamplerProjector
from ...utils import (
    LossKwargs,
    add_start_docstrings,
    logging,
    replace_return_docstrings,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LongVITAConfig"

class LongVITAModel(Qwen2Model):
    config_class = LongVITAConfig

    _no_split_modules = ["Qwen2DecoderLayer", "VisionTransformer"]
    # _no_split_modules = ["Qwen2DecoderLayer", "VisualAttentionBlock"]

    def __init__(self, config: LongVITAConfig):
        super().__init__(config)

        visual_config = InternVisionConfig(**config.visual)
        self.vision_model = InternVisionModel(visual_config)
        self.vision_projection = ResamplerProjector(config, visual_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_indices: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if (past_key_values is None or len(past_key_values) == 0) and images is not None:
            image_embeds = self.vision_model(images).last_hidden_state
            # if torch.distributed.get_rank() == 0:
            #     print(f"image_embeds {image_embeds.size()}")
            assert image_embeds.shape[0] == len(images)
            fake_images = None

            image_embeds = image_embeds[:, 1:, :]
            image_embeds = self.vision_projection(image_embeds)

            # torch.set_printoptions(threshold=100_000)
            # if torch.distributed.get_rank() == 0:
            # if True:
            #     print(f"image_embeds {image_embeds.size()}")
            #     print(f"images {images.size()}")
            #     print(f"input_ids {input_ids.size()}")
            #     # print(f"input_ids {input_ids}")
            #     print(f"image_indices {image_indices.size()}")
            #     # print(f"image_indices {image_indices}")

        elif self.training:
            device = self.get_input_embeddings().weight.data.device
            dtype = self.get_input_embeddings().weight.data.dtype
            fake_images = torch.ones((1, 3, self.config.visual["image_size"], self.config.visual["image_size"]), dtype=dtype, device=device)
            image_embeds = self.vision_model(fake_images).last_hidden_state
            image_embeds = image_embeds[:, 1:, :]
            image_embeds = self.vision_projection(image_embeds)
        else:
            fake_images = None
            image_embeds = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if fake_images is not None:
            inputs_embeds = inputs_embeds + image_embeds.mean() * 0.0
        elif image_embeds is not None:
            inputs_embeds = inputs_embeds.clone()
            image_embeds = image_embeds.to(inputs_embeds.device)
            image_indices = image_indices.to(inputs_embeds.device)
            indices_b, indices_s = image_indices.unbind(dim=0)
            inputs_embeds[indices_b.view(-1), indices_s.view(-1)] = image_embeds.view(-1, image_embeds.shape[-1])
            # inputs_embeds = inputs_embeds + image_embeds.mean() * 0.0

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LongVITAForCausalLM(Qwen2ForCausalLM):
    config_class = LongVITAConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LongVITAModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_indices: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_indices=image_indices,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
