# Copyright 2025 Deepseek-AI and the HuggingFace Inc. team. All rights reserved.
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

import math
from typing import Optional, Union

import torch
from torch import nn

from transformers.models.llava_next.modeling_llava_next import LlavaNextModel

from ...configuration_utils import PreTrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..clip.modeling_clip import CLIPEncoder, CLIPVisionEmbeddings, CLIPVisionModel, CLIPVisionTransformer
from ..llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    image_size_to_num_patches,
)
from ..sam.modeling_sam import SamVisionEncoder
from .configuration_deepseek_ocr import DeepseekOcrConfig


logger = logging.get_logger(__name__)


class DeepseekOcrSAMConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_sam_vision"
    base_config_key = "sam_config"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=1e-10,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=None,
        mlp_ratio=4.0,
        output_channels=256,
        downsample_channels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes if global_attn_indexes is not None else [2, 5, 8, 11]
        self.mlp_ratio = mlp_ratio
        self.output_channels = output_channels
        self.downsample_channels = downsample_channels if downsample_channels is not None else [512, 1024]
        self.mlp_dim = int(hidden_size * mlp_ratio)
        self.out_channels = output_channels


class DeepseekOcrCLIPVisionConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_clip_vision"
    base_config_key = "clip_vision_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        projection_dim=768,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor


class DeepseekOcrProjectorConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_projector"
    base_config_key = "projector_config"

    def __init__(
        self,
        input_dim=2048,
        n_embed=1280,
        projector_type="linear",
        depth=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.projector_type = projector_type
        self.depth = depth


class DeepseekOcrConfig(PreTrainedConfig):
    model_type = "deepseek_ocr"
    sub_configs = {
        "text_config": AutoConfig,
        "sam_vision_config": DeepseekOcrSAMConfig,
        "clip_vision_config": DeepseekOcrCLIPVisionConfig,
        "projector_config": DeepseekOcrProjectorConfig,
    }

    def __init__(
        self,
        text_config=None,
        sam_vision_config=None,
        clip_vision_config=None,
        projector_config=None,
        candidate_resolutions=None,
        global_view_pos="head",
        tile_tag="2D",
        image_token_index=100015,
        **kwargs,
    ):
        if candidate_resolutions is None:
            candidate_resolutions = [[1024, 1024]]

        self.candidate_resolutions = candidate_resolutions
        self.global_view_pos = global_view_pos
        self.tile_tag = tile_tag
        self.image_token_index = image_token_index

        if sam_vision_config is None:
            self.sam_vision_config = DeepseekOcrSAMConfig()
        elif isinstance(sam_vision_config, dict):
            self.sam_vision_config = DeepseekOcrSAMConfig(**sam_vision_config)
        else:
            self.sam_vision_config = sam_vision_config

        if clip_vision_config is None:
            self.clip_vision_config = DeepseekOcrCLIPVisionConfig()
        elif isinstance(clip_vision_config, dict):
            self.clip_vision_config = DeepseekOcrCLIPVisionConfig(**clip_vision_config)
        else:
            self.clip_vision_config = clip_vision_config

        if projector_config is None:
            self.projector_config = DeepseekOcrProjectorConfig()
        elif isinstance(projector_config, dict):
            self.projector_config = DeepseekOcrProjectorConfig(**projector_config)
        else:
            self.projector_config = projector_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "deepseek_v2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["deepseek_v2"](
                hidden_size=1280,
                intermediate_size=6848,
                num_hidden_layers=12,
                num_attention_heads=10,
                num_key_value_heads=10,
                moe_intermediate_size=896,
                n_routed_experts=64,
                n_shared_experts=2,
                num_experts_per_tok=6,
                first_k_dense_replace=1,
                vocab_size=129280,
                max_position_embeddings=8192,
                use_mla=False,
            )

        self.text_config = text_config
        self.hidden_size = text_config.hidden_size
        self.vocab_size = text_config.vocab_size

        super().__init__(**kwargs)


class DeepseekOcrProjector(nn.Module):
    """
    Projector that maps concatenated SAM + CLIP features to language model space.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.projector_type == "identity":
            self.layers = nn.Identity()
        elif config.projector_type == "linear":
            self.layers = nn.Linear(config.input_dim, config.n_embed)
        elif config.projector_type == "mlp_gelu":
            mlp_depth = config.get("depth", 1)
            modules = [nn.Linear(config.input_dim, config.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed, config.n_embed))
            self.layers = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

    def forward(self, x):
        return self.layers(x)


class DeepseekOcrSAMVisionEncoder(SamVisionEncoder):
    """
    SAM ViT-B vision encoder with additional neck layers for Deepseek OCR.
    Wraps the SAM vision encoder and adds downsampling convolutions.
    """

    def __init__(self, config):
        super().__init__()
        out_channels = config.out_channels
        downsample_channels = config.downsample_channels

        # TODO move hardcoded values to config
        self.net_2 = nn.Conv2d(out_channels, downsample_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(
            downsample_channels[0], downsample_channels[1], kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, pixel_values):
        encoder_output = self.encoder(pixel_values)
        hidden_states = encoder_output.last_hidden_state

        hidden_states = self.net_2(hidden_states)
        hidden_states = self.net_3(hidden_states)

        return hidden_states


class DeepseekOcrVisionEmbeddings(CLIPVisionEmbeddings):
    def forward(self, pixel_values, patch_embeds, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape

        # if patch_embeds is not None:
        #    patch_embeds = patch_embeds
        # else:
        patch_embeds = self.patch_embedding(pixel_values)  # Deepseek OCR CLIP embedder always uses SAM features

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class DeepseekOcrCLIPEncoder(CLIPEncoder):
    pass


class DeepseekOcrCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config):
        super().__init__()
        self.embeddings = DeepseekOcrVisionEmbeddings(config)
        self.encoder = DeepseekOcrCLIPEncoder(config)

    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        patch_embeds: Optional[torch.FloatTensor] = None,  # from SAM
        interpolate_pos_encoding: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, patch_embeds, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class DeepseekOcrPreTrainedModel(PreTrainedModel):
    config_class = DeepseekOcrConfig
    base_model_prefix = "model"


class DeepseekOcrCLIPVisionModel(CLIPVisionModel):
    def __init__(self, config):
        super().__init__(config)
        self.post_init()
        self.vision_model = DeepseekOcrCLIPVisionTransformer(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @check_model_inputs(tie_last_hidden_states=False)
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        patch_embeds: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, DeepseekOcrCLIPVisionModel

        >>> model = DeepseekOcrCLIPVisionModel.from_pretrained("openai/deepseek_ocr_c_l_i_p-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/deepseek_ocr_c_l_i_p-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""

        return self.vision_model(
            pixel_values=pixel_values,
            patch_embeds=patch_embeds,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )


class DeepseekOcrModel(LlavaNextModel):
    """
    Deepseek OCR model with dual vision encoders (SAM + CLIP) and a projector.
    """

    def __init__(self, config: DeepseekOcrConfig):
        super().__init__(config)
        self.config = config

        self.language_model = AutoModel.from_config(config.text_config)

        self.sam_model = DeepseekOcrSAMVisionEncoder(config.sam_vision_config)
        self.clip_model = DeepseekOcrCLIPVisionModel(config.clip_vision_config)

        self.projector = DeepseekOcrProjector(config.projector_config)

        embed_std = 1 / math.sqrt(config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.hidden_size) * embed_std)
        self.view_seperator = nn.Parameter(
            torch.randn(config.hidden_size) * embed_std
        )  # TODO the typo is in the checkpoint

        self.post_init()

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_token_id):
        if input_ids is None:
            tok_embed = self.get_input_embeddings()(torch.tensor(image_token_id, device=inputs_embeds.device))
            mask = (inputs_embeds == tok_embed).all(dim=-1)
        else:
            mask = input_ids == self.config.image_token_id
        return mask.unsqueeze(-1).expand_as(inputs_embeds)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,  # (B, num_patches, 3, H, W) or (sum_patches, 3, H, W)
        image_sizes: torch.Tensor,  # (num_images, 2) actual (H, W)
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ):
        patch = self.config.vision_config.patch_size
        image_num_patches = [
            image_size_to_num_patches(imsize, self.config.image_grid_pinpoints, patch) for imsize in image_sizes
        ]

        if pixel_values.dim() == 5:
            per_img = [pv[:n] for pv, n in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(per_img, dim=0)
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values has shape {pixel_values.shape}, expected 4D or 5D")

        sam_features = self.sam_model(pixel_values)
        sam_seq = sam_features.flatten(2).permute(0, 2, 1)

        clip_out = self.clip_model(pixel_values, sam_features)

        vision_feature_layer_index = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )

        if isinstance(vision_feature_layer_index, int):
            clip_seq = clip_out.hidden_states[vision_feature_layer_index]
        else:
            pool = [clip_out.hidden_states[i] for i in vision_feature_layer_index]
            clip_seq = torch.cat(pool, dim=-1)

        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        if vision_feature_select_strategy == "default":
            clip_seq = clip_seq[:, 1:]
        elif vision_feature_select_strategy != "full":
            raise ValueError(f"Unexpected vision_feature_select_strategy={vision_feature_select_strategy}")

        fused = torch.cat([clip_seq, sam_seq], dim=-1)
        proj = self.multi_modal_projector(fused)

        proj_list = torch.split(proj, image_num_patches, dim=0)

        new_image_features, _ = self.pack_image_features(
            image_features=proj_list,
            image_sizes=image_sizes,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_newline=self.image_newline,
        )

        new_image_features = [torch.cat([pf, self.view_seperator[None].to(pf)], dim=0) for pf in new_image_features]
        return torch.cat(new_image_features, dim=0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_spatial_crop: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and torch.sum(pixel_values[0, 1]).item() != 0:
            vision_features = self.get_image_features(pixel_values, image_spatial_crop)

            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, vision_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, vision_features.to(inputs_embeds.dtype))

        return self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
            **kwargs,
        )


@auto_docstring(
    custom_intro="""
    The Deepseek-OCR model which consists of two vision backbones and a deepseek language model.
    """
)
class DeepseekOcrForConditionalGeneration(LlavaNextForConditionalGeneration):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekOcrModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_spatial_crop: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_spatial_crop=image_spatial_crop,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "DeepseekOcrModel",
    "DeepseekOcrForConditionalGeneration",
    "DeepseekOcrPreTrainedModel",
    "DeepseekOcrProjector",
    "DeepseekOcrSAMVisionEncoder",
    "DeepseekOcrCLIPVisionModel",
]
