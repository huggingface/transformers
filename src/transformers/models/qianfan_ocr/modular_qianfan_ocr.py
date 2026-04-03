# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, torch_compilable_check
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..auto import AutoModel
from ..internvl.modeling_internvl import (
    NORM2FN,
    InternVLCausalLMOutputWithPast,
    InternVLModelOutputWithPast,
    InternVLMultiModalProjector,
    InternVLVisionAttention,
    InternVLVisionLayer,
    InternVLVisionMLP,
)
from .configuration_qianfan_ocr import QianfanOCRConfig, QianfanOCRVisionConfig


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class QianfanOCRVisionAttention(InternVLVisionAttention):
    pass


class QianfanOCRVisionMLP(InternVLVisionMLP):
    pass


class QianfanOCRVisionLayer(InternVLVisionLayer):
    """Vision transformer layer with stochastic depth (DropPath) support."""

    def __init__(self, config: QianfanOCRVisionConfig, drop_path_rate: float = 0.0) -> None:
        super().__init__(config)
        self.attention = QianfanOCRVisionAttention(config)
        self.mlp = QianfanOCRVisionMLP(config)
        self.layernorm_before = NORM2FN[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = NORM2FN[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        init_values = config.layer_scale_init_value
        self.lambda_1 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
        self.lambda_2 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.drop_path1 = nn.Identity() if drop_path_rate <= 0.0 else DropPath(drop_path_rate)
        self.drop_path2 = nn.Identity() if drop_path_rate <= 0.0 else DropPath(drop_path_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        attention_output, _ = self.attention(
            self.layernorm_before(hidden_states),
        )

        attention_output = self.lambda_1 * attention_output

        # first residual connection with drop path
        hidden_states = self.drop_path1(attention_output) + hidden_states

        # layernorm after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.mlp(layer_output)
        layer_output = self.dropout(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection with drop path
        layer_output = self.drop_path2(layer_output) + hidden_states

        return layer_output


class QianfanOCRVisionEncoder(nn.Module):
    def __init__(self, config: QianfanOCRVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        n = config.num_hidden_layers
        rate = float(config.drop_path_rate)
        if n <= 1:
            dpr = [0.0] * n
        else:
            dpr = [rate * i / (n - 1) for i in range(n)]

        self.layer = nn.ModuleList([QianfanOCRVisionLayer(config, drop_path_rate=dpr[i]) for i in range(n)])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple | BaseModelOutput:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class QianfanOCRVisionEmbeddings(nn.Module):
    def __init__(self, config: QianfanOCRVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        image_size = config.image_size
        patch_size = config.patch_size
        if isinstance(image_size, (list, tuple)):
            self.image_size_h, self.image_size_w = image_size[0], image_size[1]
        else:
            self.image_size_h = self.image_size_w = image_size
        if isinstance(patch_size, (list, tuple)):
            self.patch_size_h, self.patch_size_w = patch_size[0], patch_size[1]
        else:
            self.patch_size_h = self.patch_size_w = patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=(self.patch_size_h, self.patch_size_w),
            stride=(self.patch_size_h, self.patch_size_w),
        )
        self.num_patches = (self.image_size_h // self.patch_size_h) * (self.image_size_w // self.patch_size_w)
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed: torch.Tensor, H: int, W: int) -> torch.Tensor:
        target_dtype = pos_embed.dtype
        base_h = self.image_size_h // self.patch_size_h
        base_w = self.image_size_w // self.patch_size_w
        pos_embed = pos_embed.float().reshape(1, base_h, base_w, -1).permute(0, 3, 1, 2)
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor, bool_masked_pos=None, **kwargs) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(target_dtype))
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            dim=1,
        )
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class QianfanOCRVisionModelOutputWithPooling(BaseModelOutputWithPooling):
    r"""
    pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
        *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
        will be returned.
    """
    pass


@auto_docstring
class QianfanOCRVisionPreTrainedModel(PreTrainedModel):
    config_class = QianfanOCRVisionConfig
    base_model_prefix = "vision_model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QianfanOCRVisionLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": OutputRecorder(QianfanOCRVisionLayer, index=0),
        "attentions": OutputRecorder(QianfanOCRVisionAttention, index=1),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, QianfanOCRVisionEmbeddings):
            nn.init.zeros_(module.class_embedding)
            nn.init.zeros_(module.position_embedding)
        elif isinstance(module, QianfanOCRVisionLayer):
            nn.init.constant_(module.lambda_1, self.config.layer_scale_init_value)
            nn.init.constant_(module.lambda_2, self.config.layer_scale_init_value)


@auto_docstring
class QianfanOCRVisionModel(QianfanOCRVisionPreTrainedModel):
    def __init__(self, config: QianfanOCRVisionConfig) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = QianfanOCRVisionEmbeddings(config)
        self.encoder = QianfanOCRVisionEncoder(config)

        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embedding

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, bool_masked_pos: torch.BoolTensor | None = None, **kwargs
    ) -> tuple | QianfanOCRVisionModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        return QianfanOCRVisionModelOutputWithPooling(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class QianfanOCRMultiModalProjector(InternVLMultiModalProjector):
    def __init__(self, config: QianfanOCRConfig):
        super().__init__(config)


class QianfanOCRPreTrainedModel(PreTrainedModel):
    config_class = QianfanOCRConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _no_split_modules = ["QianfanOCRVisionLayer", "Qwen3DecoderLayer"]


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for QianfanOCR outputs, with hidden states and attentions.
    """
)
class QianfanOCRModelOutputWithPast(InternVLModelOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """


class QianfanOCRModel(QianfanOCRPreTrainedModel):
    def __init__(self, config: QianfanOCRConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = QianfanOCRMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if not torch.compiler.is_compiling():
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                lambda: f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {n_image_features}",
            )
        return special_image_mask

    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):
        """Perform pixel shuffle downsampling on vision features.

        Args:
            vision_features (`torch.Tensor`):
                Input tensor of shape (batch_size, width, height, channels).
            scale_factor (`float`, *optional*, defaults to `0.5`):
                Factor by which to downsample.

        Returns:
            vision_features (`torch.Tensor`):
                Downsampled tensor.
        """
        batch_size, width, height, channels = vision_features.size()

        vision_features = vision_features.view(
            batch_size, width, int(height * scale_factor), int(channels / scale_factor)
        )
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        vision_features = vision_features.view(
            batch_size, int(height * scale_factor), int(width * scale_factor), int(channels / (scale_factor**2))
        )

        if self.config.ps_version == "v1":
            import warnings

            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        return vision_features

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision tower and apply multimodal projection."
    )
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
            The tensors corresponding to the input images.
        vision_feature_layer (`int` or `list[int]`):
            Layer index or list of layer indices to extract features from.
        """
        # Use pixel_values dtype as fallback: under nn.DataParallel a replica may
        # have no parameters on the primary device (self.dtype raises StopIteration),
        # but input tensors are always correctly scattered so pixel_values.dtype is safe.

        downsample_ratio = self.config.downsample_ratio
        if vision_feature_layer != -1:
            kwargs["output_hidden_states"] = True
        vision_outputs = self.vision_tower(pixel_values=pixel_values, return_dict=True, **kwargs)
        if vision_feature_layer == -1:
            vision_features = vision_outputs.last_hidden_state
        else:
            vision_features = vision_outputs.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            vision_features = vision_features[:, 1:, :]

        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]

        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)
        vision_features = self.pixel_shuffle(vision_features, scale_factor=downsample_ratio)
        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])

        vision_features = self.multi_modal_projector(vision_features)
        vision_outputs.pooler_output = vision_features

        return vision_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | QianfanOCRModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                return_dict=True,
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return QianfanOCRModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for QianfanOCR causal language model outputs.
    """
)
class QianfanOCRCausalLMOutputWithPast(InternVLCausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """


class QianfanOCRForConditionalGeneration(QianfanOCRPreTrainedModel, GenerationMixin):
    def __init__(self, config: QianfanOCRConfig):
        super().__init__(config)
        self.model = QianfanOCRModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_sizes: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | QianfanOCRCausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText

        >>> torch_device = "cuda"
        >>> processor = AutoProcessor.from_pretrained("baidu/Qianfan-OCR")
        >>> model = AutoModelForImageTextToText.from_pretrained(
        ...     "baidu/Qianfan-OCR", dtype=torch.bfloat16, device_map=torch_device
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "url": "https://example.com/image.jpg"},
        ...             {"type": "text", "text": "Describe this image."},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device)
        >>> generate_ids = model.generate(**inputs, max_new_tokens=200)
        >>> print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=image_sizes,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return QianfanOCRCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration or not kwargs.get("use_cache", True):
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


__all__ = [
    "QianfanOCRVisionPreTrainedModel",
    "QianfanOCRVisionModel",
    "QianfanOCRPreTrainedModel",
    "QianfanOCRModel",
    "QianfanOCRForConditionalGeneration",
]
