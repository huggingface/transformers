# Copyright 2026 NAVER Corp. and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HyperCLOVAX Vision model."""

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub, use_kernelized_func
from ...masking_utils import create_causal_mask
from ...modeling_layers import GenericForSequenceClassification
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.auto import AutoModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from ..qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLModelOutputWithPast
from .configuration_hyperclovax_vision import HCXVisionConfig, HyperClovaXConfig


logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class HyperCLOVAXRMSNorm(LlamaRMSNorm):
    pass


class HyperCLOVAXRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class HyperCLOVAXMLP(LlamaMLP):
    pass


@use_kernelized_func(apply_rotary_pos_emb)
class HyperCLOVAXAttention(LlamaAttention):
    pass


class HyperCLOVAXDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HyperClovaXConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.use_post_norm = config.use_post_norm
        if self.use_post_norm:
            self.post_norm1 = HyperCLOVAXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_norm2 = HyperCLOVAXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.residual_multiplier = config.residual_multiplier  # mup

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        if self.use_post_norm:
            hidden_states = self.post_norm1(hidden_states)

        hidden_states = residual + hidden_states * self.residual_multiplier  # mup

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.use_post_norm:
            hidden_states = self.post_norm2(hidden_states)

        hidden_states = residual + hidden_states * self.residual_multiplier  # mup
        return hidden_states


@auto_docstring
class HCXVisionPreTrainedModel(PreTrainedModel):
    config_class = HCXVisionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HyperCLOVAXDecoderLayer", "Qwen2_5_VLVisionBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    input_modalities = ("image", "video", "text")
    _can_compile_fullgraph = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": HyperCLOVAXDecoderLayer,
        "attentions": HyperCLOVAXAttention,
    }

    def _init_weights(self, module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, (nn.LayerNorm, HyperCLOVAXRMSNorm)):
            init.ones_(module.weight)
            if hasattr(module, "bias"):
                init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            embed_std = 1 / torch.sqrt(torch.tensor(module.size(0), dtype=torch.float)).to(module.dtype)
            init.normal_(module, mean=0.0, std=embed_std)

        if isinstance(module, HyperCLOVAXRotaryEmbedding):
            inv_freq, _ = module.compute_default_rope_parameters(module.config)
            init.copy_(module.inv_freq, inv_freq)
            init.copy_(module.original_inv_freq, inv_freq)


@auto_docstring
class HyperClovaXTextModel(HCXVisionPreTrainedModel):
    config_class = HyperClovaXConfig
    input_modalities = ("text",)

    def __init__(self, config: HyperClovaXConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HyperCLOVAXDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HyperCLOVAXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HyperCLOVAXRotaryEmbedding(config=config)
        self.embedding_multiplier = config.embedding_multiplier  # mup
        self.gradient_checkpointing = False

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        inputs_embeds = inputs_embeds * self.embedding_multiplier  # mup
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class HyperClovaXForCausalLM(HCXVisionPreTrainedModel, GenerationMixin):
    accepts_loss_kwargs = False
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = HyperClovaXConfig
    input_modalities = ("text",)

    def __init__(self, config: HyperClovaXConfig) -> None:
        super().__init__(config)
        self.model = HyperClovaXTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in
            `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices
            set to `-100` are ignored (masked), the loss is only computed for the tokens with labels
            in `[0, ..., config.vocab_size]`.
        logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate
            logits for all `input_ids`. Only last token logits are needed for generation, and
            computing them only for that token can save memory for long sequences or large vocabulary sizes.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, HyperClovaXForCausalLM

        >>> model = HyperClovaXForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
        >>> tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

        >>> prompt = "안녕하세요! 오늘 날씨가 어떻습니까?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> generate_ids = model.generate(inputs.input_ids, max_new_tokens=50)
        >>> tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        ```
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits * self.config.logits_scaling  # mup

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HCXVisionMultiModalProjector(nn.Module):
    def __init__(self, config: HCXVisionConfig) -> None:
        super().__init__()

        vision_config = config.vision_config
        text_config = config.text_config

        input_hidden_size = getattr(vision_config, "out_hidden_size", vision_config.hidden_size)
        output_hidden_size = text_config.hidden_size

        self.proj = nn.Linear(input_hidden_size, output_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project vision features to the text decoder hidden space.

        Args:
            hidden_states: Vision encoder output tensor.
                For most projector types: ``(batch_size, seq_len, hidden_size)``
                or ``(seq_len, hidden_size)`` for packed sequences (qwen_merger).
                For qwen_merger with window reordering: ``tuple(hidden_states, window_index)``.

        Returns:
            Projected features with last dimension equal to ``text_config.hidden_size``.
        """

        return self.proj(hidden_states)


@auto_docstring
class HCXVisionModel(HCXVisionPreTrainedModel):
    def __init__(self, config: HCXVisionConfig) -> None:
        super().__init__(config)

        if config.vision_config.architectures[0] == "Qwen2_5_VisionTransformerPretrainedModel":
            vision_config = Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)
        else:
            vision_config = AutoModel.from_config(config.vision_config)
        self.vision_model = vision_config

        self.multi_modal_projector = HCXVisionMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        video_output = self.vision_model(pixel_values_videos, grid_thw=video_grid_thw, return_dict=True, **kwargs)
        projected = self.multi_modal_projector(video_output.pooler_output)
        video_output.pooler_output = projected
        return video_output

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        image_output = self.vision_model(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        projected = self.multi_modal_projector(image_output.pooler_output)
        image_output.pooler_output = projected
        return image_output

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.img_start_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_start_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.img_start_id
            special_video_mask = input_ids == self.config.video_start_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                inputs_embeds[special_video_mask].numel() == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )
        return special_image_mask, special_video_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor`, *optional*):
            Pixel values of input images after preprocessing by [`Qwen2VLImageProcessor`].
            A 2D tensor of shape `(total_num_patches, channels * patch_size^2 * temporal_patch_size)`.
            In the input token sequence, each image position should contain `config.img_start_id`.
        pixel_values_videos (`torch.FloatTensor`, *optional*):
            Pixel values of input videos, with the same format as `pixel_values`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width dimensions of the feature grid for each image.
            Each row contains `[temporal, height, width]` grid counts.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width dimensions of the feature grid for each video.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=None,
        )


@auto_docstring
class HCXVisionForConditionalGeneration(HCXVisionPreTrainedModel, GenerationMixin):
    accepts_loss_kwargs = False
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: HCXVisionConfig):
        super().__init__(config)
        self.model = HCXVisionModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor`, *optional*):
            Pixel values of input images after preprocessing by [`Qwen2VLImageProcessor`].
            For any-resolution mode, this is a 2D tensor of shape
            `(total_num_patches, channels * patch_size^2 * temporal_patch_size)` where
            `total_num_patches` is the sum of patches across all images in the batch.
            In the input token sequence, each image position should contain `config.img_start_id`.
        pixel_values_videos (`torch.FloatTensor`, *optional*):
            Pixel values of input videos, with the same format as `pixel_values`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width dimensions of the feature grid for each image.
            Each row contains `[temporal, height, width]` grid counts.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width dimensions of the feature grid for each video.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in
            `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices
            set to `-100` are ignored (masked), the loss is only computed for the tokens with labels
            in `[0, ..., config.vocab_size]`.
        logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate
            logits for all `input_ids`. Only last token logits are needed for generation, and
            computing them only for that token saves memory for long sequences.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, HCXVisionForConditionalGeneration

        >>> model = HCXVisionForConditionalGeneration.from_pretrained(
        ...     "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
        ...     torch_dtype="auto",
        ...     device_map="auto",
        ... )
        >>> processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

        >>> messages = [
        ...     {"role": "user", "content": [
        ...         {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
        ...         {"type": "text", "text": "Describe this image in detail."},
        ...     ]}
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, return_dict=True, return_tensors="pt"
        ... ).to(model.device)
        >>> output = model.generate(**inputs, max_new_tokens=200)
        >>> processor.decode(output[0], skip_special_tokens=True)
        ```
        """

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]) * getattr(
            self.config.text_config, "logits_scaling", 1
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)

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
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


class HCXVisionForSequenceClassification(HCXVisionPreTrainedModel, GenericForSequenceClassification):
    accepts_loss_kwargs = False


__all__ = [
    "HCXVisionForConditionalGeneration",
    "HCXVisionForSequenceClassification",
    "HCXVisionModel",
    "HCXVisionPreTrainedModel",
    "HyperClovaXForCausalLM",
    "HyperClovaXTextModel",
]
