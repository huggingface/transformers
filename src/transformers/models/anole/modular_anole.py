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
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Tuple, Union

import numpy as np

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...image_processing_utils import BatchFeature
from ...image_transforms import (
    to_channel_dimension_format,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    make_flat_list_of_images,
    to_numpy_array,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from ...utils import TensorType, add_start_docstrings, is_torch_available, is_vision_available, logging
from ..chameleon.configuration_chameleon import (
    ChameleonConfig,
    ChameleonVQVAEConfig,
)
from ..chameleon.image_processing_chameleon import ChameleonImageProcessor
from ..chameleon.modeling_chameleon import (
    ChameleonImageVocabularyMapping,
    ChameleonModel,
    ChameleonPreTrainedModel,
    ChameleonVQVAE,
    ChameleonVQVAEEncoderAttnBlock,
    ChameleonVQVAEEncoderResnetBlock,
    ChameleonVQVAEVectorQuantizer,
)
from ..chameleon.processing_chameleon import ChameleonProcessor


if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint

if is_vision_available():
    import PIL


_CHECKPOINT_FOR_DOC = "leloy/Anole-7b-v0.1-hf"

logger = logging.get_logger(__name__)


class AnoleVQVAEConfig(ChameleonVQVAEConfig):
    def __init__(
        self,
        out_channels: int = 3,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.out_channels = out_channels


class AnoleConfig(ChameleonConfig):
    pass


@dataclass
class AnoleVQVAEOutput(ModelOutput):
    """
    Base class for Anole VQ-VAE mode model outputs.
    Args:
        decoded_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            Reconstructed pixel values after encoding and decoding the input.
        emb_loss (`torch.FloatTensor`):
            Embedding loss.
    """

    decoded_pixel_values: Optional[torch.FloatTensor] = None
    emb_loss: torch.FloatTensor = None


class AnoleVQVAEVectorQuantizer(ChameleonVQVAEVectorQuantizer):
    def __init__(self, config):
        super().__init__(config)
        self.quant_state_dims = [config.resolution // 2 ** (len(config.channel_multiplier) - 1)] * 2

    def get_codebook_entry(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        batch_size = image_tokens.shape[0]
        emb_dim: int = self.embedding.weight.shape[-1]
        # get quantized latent vectors
        hidden_state_quant = self.embedding(image_tokens)

        # reshape back to match original input shape
        hidden_state_quant = hidden_state_quant.view((batch_size, *self.quant_state_dims, emb_dim))
        hidden_state_quant = hidden_state_quant.permute(0, 3, 1, 2).contiguous()

        return hidden_state_quant


class AnoleVQVAEResnetBlock(ChameleonVQVAEEncoderResnetBlock):
    pass


class AnoleVQVAEAttnBlock(ChameleonVQVAEEncoderAttnBlock):
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


class AnolePreTrainedModel(ChameleonPreTrainedModel):
    pass


class AnoleVQVAE(ChameleonVQVAE):
    main_input_name = "pixel_values"

    def __init__(self, config: AnoleVQVAEConfig):
        super().__init__(config)
        self.decoder = AnoleVQVAEDecoder(config)
        self.gradient_checkpointing = False
        self.post_init()

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

        batch_size = pixel_values.shape[0]
        quant, emb_loss, indices = self.encode(pixel_values)
        decoded_pixel_values = self.decode(indices.view(batch_size, -1))
        if not return_dict:
            return (decoded_pixel_values, emb_loss)
        return AnoleVQVAEOutput(decoded_pixel_values, emb_loss)


class AnoleImageVocabularyMapping(ChameleonImageVocabularyMapping):
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

    @property
    def image_seq_length(self) -> int:
        return self.vqmodel.quantize.quant_state_dims[0] * self.vqmodel.quantize.quant_state_dims[1]

    def convert_img2bpe_tokens(self, img_batch: torch.LongTensor) -> torch.LongTensor:
        """
        Converts image tokens generated by the VQVAE model into BPE tokens compatible with the text tokenizer.

        Notes:
            - It is important to move the `img_batch` tensor to the same device as the `img2bpe_mapping_tensor` buffer
            as Accelerate may move the buffer to a different device when loading the model with `device_map="auto"`.
            - Accelerate up to version 0.33.0 (and also maybe later versions) has a bug where buffers in downstream modules
            may be ignored when inferring the proper device map. See: https://github.com/huggingface/accelerate/blob/79ca85c27df292dbf64cfa2bcc12dbb62fbe9267/src/accelerate/utils/modeling.py#L1273
            This causes the `img2bpe_mapping_tensor` buffer to be placed on the CPU by default, which may cause a performance
            loss--especially with prompts that contain many images. No action needs to be done when this bug is fixed.

        Args:
            img_batch (`torch.Tensor` of shape `(batch_size, image_seq_length)`):
                The image tokens generated by the VQVAE model.

        Returns:
            `torch.Tensor` of shape `(batch_size, image_seq_length)`:
                The image tokens converted to be compatible with the text tokenizer's BPE tokens.
        """
        device = img_batch.device
        img_tokens = self.img2bpe_mapping_tensor[img_batch.to(self.img2bpe_mapping_tensor.device)]
        return img_tokens.to(device)

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

    def get_image_tokens(self, pixel_values: torch.FloatTensor):
        """
        Tokenizes images into discrete tokens with VQGAN module. Converts
        obtained image tokens into BPE tokens and wraps with "boi" and "eoi"
        special tokens.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
                The tensors corresponding to the input images.

        Returns:
            `torch.Tensor` of shape `(batch_size, image_seq_length)`:
                The BPE tokens generated by the model.
        """
        batch_size = pixel_values.shape[0]
        _, _, image_toks = self.vqmodel.encode(pixel_values)
        image_toks = image_toks.view(batch_size, -1)
        return self.convert_img2bpe_tokens(image_toks)

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


ANOLE_START_DOCSTRING = None


@add_start_docstrings(
    "Anole Model with a head on top used for outputting logits for next token prediction.",
    ANOLE_START_DOCSTRING,
)
class AnoleForConditionalGeneration(AnolePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: AnoleConfig):
        super().__init__(config)
        self.model = AnoleModel(config)
        self.vocabulary_mapping = self.model.vocabulary_mapping
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
    def postprocess(
        self,
        images: ImageInput,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_unnormalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> "torch.Tensor":
        """
        Postprocess a batch of pixel values to images.

        Args:
            images (`ImageInput`):
                Image to postprocess. Expects a single or batch of images with pixel values.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_unnormalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to unnormalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for unnormalization. Only has an effect if `do_unnormalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for unnormalization. Only has an effect if `do_unnormalize` is set to
                `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `PIL.Image` or `'pil'`: Return a batch of type `PIL.Image`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = 1.0 / self.rescale_factor if rescale_factor is None else rescale_factor
        do_unnormalize = do_unnormalize if do_unnormalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_flat_list_of_images(images)

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        postprocessed_images = []
        for image in images:
            if do_unnormalize:
                image = self.unnormalize(
                    image, mean=image_mean, std=image_std, data_format=data_format, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)
                image = image.clip(0, 255).astype(np.uint8)

            if do_unnormalize and do_rescale and return_tensors == "pil":
                image = to_channel_dimension_format(image, ChannelDimension.LAST, input_channel_dim=input_data_format)
                image = PIL.Image.fromarray(image)
            postprocessed_images.append(image)

        return_tensors = return_tensors if return_tensors != "pil" else None
        return BatchFeature(data={"pixel_values": postprocessed_images}, tensor_type=return_tensors)


class AnoleProcessor(ChameleonProcessor):
    def postprocess(self, images, return_tensors=None, **kwargs):
        """
        Postprocess a batch of images.

        Args:
            images (`ImageInput`):
                A batch of images or a single image to postprocess.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values that are postprocessed in the requested return type.
        """
        return self.image_processor.postprocess(images, return_tensors=return_tensors, **kwargs)


__all__ = [
    "AnoleConfig",
    "AnoleVQVAEConfig",
    "AnoleVQVAE",
    "AnoleModel",
    "AnoleForConditionalGeneration",
    "AnoleImageProcessor",
    "AnoleProcessor",
]
