# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
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
from functools import cached_property
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...utils import is_torchdynamo_compiling, logging
from ..chameleon.configuration_chameleon import (
    ChameleonConfig,
    ChameleonVQVAEConfig,
)
from ..chameleon.image_processing_chameleon import ChameleonImageProcessor
from ..chameleon.modeling_chameleon import (
    ChameleonForConditionalGeneration,
    ChameleonImageVocabularyMapping,
    ChameleonModel,
    ChameleonVQVAE,
    ChameleonVQVAEAttnBlock,
    ChameleonVQVAEOutput,
    ChameleonVQVAEResnetBlock,
)
from ..chameleon.processing_chameleon import ChameleonProcessor


_CHECKPOINT_FOR_DOC = "leloy/Anole-7b-v0.1-hf"

logger = logging.get_logger(__name__)


class AnoleVQVAEConfig(ChameleonVQVAEConfig):
    pass


class AnoleConfig(ChameleonConfig):
    pass


class AnoleVQVAEOutput(ChameleonVQVAEOutput):
    pass


class AnoleVQVAEResnetBlock(ChameleonVQVAEResnetBlock):
    pass


class AnoleVQVAEAttnBlock(ChameleonVQVAEAttnBlock):
    pass


class AnoleVQVAEDecoderConvUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states


class AnoleVQVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        resolution = config.resolution
        latent_channels = config.latent_channels
        out_channels = config.out_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = base_channels * config.channel_multiplier[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, latent_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(latent_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = AnoleVQVAEResnetBlock(
            config=config,
            in_channels=block_in,
            out_channels=block_in,
        )
        self.mid.attn_1 = AnoleVQVAEAttnBlock(block_in) if config.attn_type == "vanilla" else nn.Identity()
        self.mid.block_2 = AnoleVQVAEResnetBlock(
            config=config,
            in_channels=block_in,
            out_channels=block_in,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = base_channels * config.channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    AnoleVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if (
                    config.attn_resolutions is not None
                    and curr_res in config.attn_resolutions
                    and config.attn_type == "vanilla"
                ):
                    attn.append(AnoleVQVAEAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = AnoleVQVAEDecoderConvUpsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        hidden_state = self.conv_in(hidden_state)

        # middle
        hidden_state = self.mid.block_1(hidden_state)
        hidden_state = self.mid.attn_1(hidden_state)
        hidden_state = self.mid.block_2(hidden_state)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden_state = self.up[i_level].block[i_block](hidden_state)
                if len(self.up[i_level].attn) > 0:
                    hidden_state = self.up[i_level].attn[i_block](hidden_state)
            if i_level != 0:
                hidden_state = self.up[i_level].upsample(hidden_state)

        hidden_state = self.norm_out(hidden_state)
        hidden_state *= torch.sigmoid(hidden_state)
        hidden_state = self.conv_out(hidden_state)
        return hidden_state


class AnoleVQVAE(ChameleonVQVAE):
    def __init__(self, config: AnoleVQVAEConfig):
        super().__init__(config)
        self.decoder = AnoleVQVAEDecoder(config)
        self.quant_conv = torch.nn.Conv2d(config.latent_channels, config.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(config.embed_dim, config.latent_channels, 1)

    def decode(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Decodes quantized token IDs into pixel values.

        Args:
            image_tokens (`torch.LongTensor` of shape `(batch_size, quantize.quant_state_dims[0] * quantize.quant_state_dims[1])`):
                Batch of token IDs.

        Returns:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                Pixel values decoded from the token IDs.
        """
        if image_tokens.shape[1] != self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]:
            raise ValueError(
                f"Expected `image_tokens` to have shape `(batch_size, {self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]})`, "
                f"but got shape `{image_tokens.shape}`."
            )
        codebook_entry = self.quantize.get_codebook_entry(image_tokens)
        hidden_states = self.post_quant_conv(codebook_entry)
        pixel_values = self.decoder(hidden_states)
        return pixel_values

    def forward(
        self, pixel_values: torch.FloatTensor, return_dict: bool = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Encodes pixel values into quantized tokens and decodes them back.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
                The tensors corresponding to the input images.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            decoded_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                Reconstructed pixel values after encoding and decoding the input.
            emb_loss (`torch.FloatTensor`):
                Embedding loss.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        quant, emb_loss, indices = self.encode(pixel_values)
        decoded_pixel_values = self.decode(indices)
        if not return_dict:
            return (decoded_pixel_values, emb_loss)
        return AnoleVQVAEOutput(decoded_pixel_values, emb_loss)


class AnoleImageVocabularyMapping(ChameleonImageVocabularyMapping):
    @cached_property
    def bpe2img(self):
        img_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}

        def remap(old_name: str) -> str:
            return "".join(img_tkn_chr_mapping.get(c, c) for c in old_name[len("IMGIMG") : -1])

        return {tok: int(remap(self.val2name[tok])) for tok in self.image_token_ids}

    @cached_property
    def bpe2img_mapping_tensor(self):
        mapping = torch.zeros(max(self.bpe2img.keys()) + 1, dtype=torch.int)
        for k, v in self.bpe2img.items():
            mapping[k] = v
        return mapping


class AnoleModel(ChameleonModel):
    def __init__(self, config: AnoleConfig):
        super().__init__(config)
        self.register_buffer(
            "img2bpe_mapping_tensor",
            self.vocabulary_mapping.img2bpe_mapping_tensor,
            persistent=False,
        )
        self.register_buffer(
            "bpe2img_mapping_tensor",
            self.vocabulary_mapping.bpe2img_mapping_tensor,
            persistent=False,
        )

    property

    def image_seq_length(self) -> int:
        return self.vqmodel.quantize.quant_state_dims[0] * self.vqmodel.quantize.quant_state_dims[1]

    def convert_bpe2img_tokens(self, bpe_batch: torch.LongTensor) -> torch.LongTensor:
        """
        Converts image tokens that are compatible with the text tokenizer into image tokens compatible with the VQVAE
        model.

        Notes:
            - It is important to move the `img_batch` tensor to the same device as the `img2bpe_mapping_tensor` buffer
            as Accelerate may move the buffer to a different device when loading the model with `device_map="auto"`.
            - Accelerate up to version 0.33.0 (and also maybe later versions) has a bug where buffers in downstream modules
            may be ignored when inferring the proper device map. See: https://github.com/huggingface/accelerate/blob/79ca85c27df292dbf64cfa2bcc12dbb62fbe9267/src/accelerate/utils/modeling.py#L1273
            This causes the `img2bpe_mapping_tensor` buffer to be placed on the CPU by default, which may cause a performance
            loss--especially when generating interleaved text & images. No action needs to be done when this bug is fixed.

        Args:
            bpe_batch (`torch.Tensor` of shape `(batch_size, image_seq_length)`):
                The image tokens compatible with the text tokenizer.

        Returns:
            `torch.Tensor` of shape `(batch_size, image_seq_length)`:
                The image tokens converted to be compatible with the VQVAE model.
        """
        device = bpe_batch.device
        img_tokens = self.bpe2img_mapping_tensor[bpe_batch.to(self.bpe2img_mapping_tensor.device)]
        return img_tokens.to(device)

    def decode_image_tokens(self, bpe_tokens: torch.LongTensor) -> torch.LongTensor:
        """
        Converts BPE tokens generated by the model into discrete image tokens
        compatible with the VQGAN module, then decodes them into pixel values.

        Args:
            bpe_tokens (`torch.tensor` of shape `(batch, image_seq_length)`):
                The BPE tokens generated by the model.

        Returns:
            `torch.Tensor` of shape `(batch, num_channels, 512, 512)`:
        """
        if bpe_tokens.shape[1] != self.image_seq_length:
            raise ValueError(f"All batches must have {self.image_seq_length} tokens.")
        image_tensor = self.convert_bpe2img_tokens(bpe_tokens)
        return self.vqmodel.decode(image_tensor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
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

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
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
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class AnoleForConditionalGeneration(ChameleonForConditionalGeneration):
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, config: AnoleConfig):
        super().__init__(config)
        self.model = AnoleModel(config)
        self.vocabulary_mapping = self.model.vocabulary_mapping
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
        >>> import torch
        >>> import requests
        >>> from PIL import Image

        >>> model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16)
        >>> processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

        >>> prompt = "I used to know a lot about constellations when I was younger, but as I grew older, I forgot most of what I knew. These are the only two constellations that I really remember now.<image><image>I would like for you to tell me about 3 more constellations and give me a little bit of history about the constellation."
        >>> image = Image.open(requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw)
        >>> image_2 = Image.open(requests.get("https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg", stream=True).raw)

        >>> inputs = processor(images=[image, image_2], text=prompt, return_tensors="pt").to(model.device, torch.bfloat16)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        >>> processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        negative_input_ids=None,
        past_key_values=None,
        attention_mask=None,
        negative_attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Overwritten -- Anole can accept `negative_input_ids` and will expand inputs temporarily for CFG generation
        if negative_input_ids is not None:
            if input_ids.shape[0] != negative_input_ids.shape[0]:
                raise ValueError(
                    "`conditional` and `unconditional` inputs should have same batch size, i.e. one negative prompt per input. "
                    f"Found `input_ids.batch_size={input_ids.shape[0]}` and `negative_input_ids.batch_size={negative_input_ids.shape[0]}`"
                )
            input_ids = torch.cat([input_ids, negative_input_ids], dim=0)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, negative_attention_mask], dim=0)

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        if past_key_values is not None:
            if (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {
                "input_ids": input_ids.clone(memory_format=torch.contiguous_format)
            }  # `contiguous()` needed for compilation use cases

        # Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def decode_image_tokens(self, bpe_tokens: torch.Tensor):
        """
        Converts BPE tokens generated by the model into discrete image tokens
        compatible with the VQGAN module, then decodes them into pixel values.

        Args:
            bpe_tokens (`torch.tensor` of shape `(batch, image_seq_length)`):
                The BPE tokens generated by the model.

        Returns:
            `torch.Tensor` of shape `(batch, num_channels, 512, 512)`:
        """
        return self.model.decode_image_tokens(bpe_tokens)


class AnoleImageProcessor(ChameleonImageProcessor):
    pass


class AnoleProcessor(ChameleonProcessor):
    pass


__all__ = [
    "AnoleConfig",
    "AnoleVQVAEConfig",
    "AnoleVQVAE",
    "AnoleModel",
    "AnoleForConditionalGeneration",
    "AnoleImageProcessor",
    "AnoleProcessor",
]
