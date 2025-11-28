from typing import Optional, Tuple, List, Union
from einops import rearrange, repeat
import gc

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageOps

from ...configuration_utils import PretrainedConfig
from ...processing_utils import ProcessorMixin
from ..auto import AutoConfig, AutoModel, AutoModelForCausalLM, CONFIG_MAPPING
from ..deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from ..deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config

from ..deepseek_vl.image_processing_deepseek_vl import DeepseekVLImageProcessor
from ..deepseek_vl.processing_deepseek_vl import DeepseekVLProcessor
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ..deepseek_vl.modeling_deepseek_vl import (
    DeepseekVLPreTrainedModel,
    DeepseekVLForConditionalGeneration,
    DeepseekVLModel,
)

from ...utils import (
    auto_docstring,
    is_torch_available,
    logging,
)

if is_torch_available():
    import torch
    import torch.nn as nn

logger = logging.get_logger(__name__)


class MlpProjectorConfig(PretrainedConfig):
    model_type = "mlp_projector"
    projector_type: str = "downsample_mlp_gelu"
    input_dim: int = 1152
    n_embed: int = 2048
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False

    def __init__(
        self,
        projector_type: str = "downsample_mlp_gelu",
        input_dim: int = 1152,
        n_embed: int = 2048,
        depth: int = 2,
        mlp_ratio: int = 1,
        downsample_ratio: int = 2,
        token_pooling: bool = False,
        **kwargs,
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.downsample_ratio = downsample_ratio
        self.token_pooling = token_pooling

        super().__init__(**kwargs)


class MlpProjector(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.depth
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [
                nn.Linear(
                    cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio,
                    cfg.n_embed * mlp_ratio,
                )
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio)
                )
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        if cfg.token_pooling:
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

        self.layers = modules

    def forward(self, x):
        if self.cfg.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)

            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()

            patches = patches.contiguous().view(
                batch_size, channels, h_patches * w_patches, -1
            )

            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        elif self.cfg.projector_type == "downsample_mlp_gelu":
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(
                x,
                kernel_size=self.cfg.downsample_ratio,
                stride=self.cfg.downsample_ratio,
                padding=0,
            )
            x = x.permute(0, 2, 1)

        return self.layers(x)


class DeepseekVLV2Config(PretrainedConfig):
    model_type = "deepseek_vl_v2"
    sub_configs = {
        "language_config": AutoConfig,
        "vision_config": AutoConfig,
        "projector_config": MlpProjectorConfig,
    }

    tile_tag: str = "2D"
    global_view_pos: str = "head"
    candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384),)

    def __init__(
        self,
        tile_tag: str = "tile_tag",
        global_view_pos: str = "head",
        candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384),),
        n_embed: int = 512,
        language_config: dict = None,
        vision_config: dict = None,
        projector_config: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        self.candidate_resolutions = candidate_resolutions
        self.n_embed = n_embed

        if language_config is None:
            language_config = {}
            logger.info("`language_config` is `None`. Initializing the `DeepseekV2Config` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `SiglipVisionConfig` with default values.")

        if isinstance(language_config, dict):
            language_config["model_type"] = language_config.get("model_type", "deepseek_v2")
            language_config = CONFIG_MAPPING[language_config["model_type"]](**language_config)

        if isinstance(vision_config, dict):
            vision_config["model_type"] = "siglip_vision_model"
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)

        if isinstance(projector_config, dict):
            projector_config["model_type"] = "mlp_projector"
            projector_config = MlpProjectorConfig(**projector_config)

        self.language_config = language_config
        self.vision_config = vision_config
        self.projector_config = projector_config


class DeepseekVLV2PreTrainedModel(DeepseekVLPreTrainedModel):
    config_class = DeepseekVLV2Config
    base_model_prefix = "deepseek_vl_v2"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class DeepseekVLV2Model(DeepseekVLModel):
    def __init__(self, config: DeepseekVLV2Config):
        super().__init__(config)
        self.config = config
        self.projector = MlpProjector(config.projector_config)
        self.language_model = AutoModel.from_config(config.language_config)

        self.vision_model = AutoModel.from_config(config.vision_config)

        del self.aligner


class DeepseekVLV2ForCausalLM(DeepseekVLForConditionalGeneration):
    def __init__(self, config: DeepseekVLV2Config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_config(config.vision_config)
        self.language_model = AutoModelForCausalLM.from_config(config.language_config)

        projector_config = config.projector_config
        self.projector = MlpProjector(projector_config)

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        embed_std = 1 / torch.sqrt(
            torch.tensor(projector_config.n_embed, dtype=torch.float32)
        )

        if self.tile_tag == "2D":
            self.image_newline = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
            self.view_seperator = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
        elif self.tile_tag == "1D":
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError("candidate_resolutions should not be empty")

            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, projector_config.n_embed)) * embed_std
            )
        else:
            raise ValueError(
                f"Unknown tile_tag: {self.tile_tag}. It should one of the following: '1D', '2D'."
            )

    def get_image_features(self, tiles):
        image_features = self.vision_model(tiles)
        image_embeds = self.projector(image_features.last_hidden_state)
        return image_embeds

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        **ignore_kwargs,
    ):
        if images is None or images_spatial_crop.sum() == 0:
            return self.language_model.get_input_embeddings()(input_ids)

        bs, max_n_images, _ = images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []

        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]

                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                batch_num_tiles[idx] += 1 + num_width_tiles * num_height_tiles

            total_tiles.append(batch_num_tiles[idx])

        total_tiles = torch.cat(total_tiles, dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)

        if total_tiles.shape[0] == 0:
            return self.language_model.get_input_embeddings()(input_ids)

        images_embeds = self.get_image_features(total_tiles)
        _, hw, n_dim = images_embeds.shape
        h = w = int(hw**0.5)

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        tile_index = 0
        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]

                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles
                global_features = images_embeds[tile_index]

                local_features = images_embeds[
                    tile_index + 1 : tile_index + 1 + num_tiles_in_image
                ]

                tile_index += num_tiles_in_image + 1

                if self.tile_tag == "2D":
                    global_features = global_features.view(h, w, n_dim)
                    new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
                    global_features = torch.cat(
                        [global_features, new_lines_in_global], dim=1
                    )
                    global_features = global_features.view(-1, n_dim)

                    local_features = rearrange(
                        local_features,
                        "(th tw) (h w) d -> (th h) (tw w) d",
                        th=num_height_tiles,
                        tw=num_width_tiles,
                        h=h,
                        w=w,
                    )

                    new_lines_in_local = repeat(
                        self.image_newline, "d -> (th h) 1 d", th=num_height_tiles, h=h
                    )

                    local_features = torch.cat(
                        [local_features, new_lines_in_local], dim=1
                    )

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat(
                            [
                                global_features,
                                self.view_seperator[None, :],
                                local_features,
                            ],
                            dim=0,
                        )
                    else:
                        global_local_features = torch.cat(
                            [
                                local_features,
                                self.view_seperator[None, :],
                                global_features,
                            ],
                            dim=0,
                        )
                else:
                    global_features = torch.cat(
                        [self.tile_indicators[0:1], global_features], dim=0
                    )

                    local_features = torch.cat(
                        [
                            self.tile_indicators[1 : num_tiles_in_image + 1].unsqueeze(
                                1
                            ),
                            local_features,
                        ],
                        dim=1,
                    )

                    local_features = rearrange(
                        local_features,
                        "crop_num hw d -> (crop_num hw) d",
                    )

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat(
                            [global_features, local_features], dim=0
                        )
                    else:
                        global_local_features = torch.cat(
                            [local_features, global_features], dim=0
                        )

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)

                inputs_embeds[idx].masked_scatter_(
                    images_seq_mask[idx].unsqueeze(-1), images_in_this_batch
                )

        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.language.forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        attention_mask=None,
        cache_position=None,
        pixel_values=None,
        image_sizes=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = self.language.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        cache_position = model_inputs["cache_position"]
        if cache_position[0] == 0:
            model_inputs["images"] = images
            model_inputs["images_seq_mask"] = images_seq_mask
            model_inputs["images_spatial_crop"] = images_spatial_crop

        return model_inputs


class DeepseekVLV2ImageProcessor(DeepseekVLImageProcessor):
    def __init__(
        self,
        image_mean: Optional[Union[float, list[float]]] = [0.5, 0.5, 0.5],
        image_std: Optional[Union[float, list[float]]] = [0.5, 0.5, 0.5],
        candidate_resolutions: Optional[Tuple[Tuple[int, int]]] = None,
        patch_size: Optional[int] = None,
        downsample_ratio: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(image_mean=image_mean, image_std=image_std, **kwargs)
        self.candidate_resolutions = candidate_resolutions
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )

    def select_best_resolution(self, image_size, candidate_resolutions):
        original_width, original_height = image_size
        best_fit = None
        max_effective_resolution = 0
        min_wasted_resolution = float("inf")

        for width, height in candidate_resolutions:
            scale = min(width / original_width, height / original_height)
            downscaled_width, downscaled_height = int(original_width * scale), int(
                original_height * scale
            )
            effective_resolution = min(
                downscaled_width * downscaled_height, original_width * original_height
            )
            wasted_resolution = (width * height) - effective_resolution

            if effective_resolution > max_effective_resolution or (
                effective_resolution == max_effective_resolution
                and wasted_resolution < min_wasted_resolution
            ):
                max_effective_resolution = effective_resolution
                min_wasted_resolution = wasted_resolution
                best_fit = (width, height)

        return best_fit

    def preprocess(self, image: Image.Image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        w, h = image.size
        best_w, best_h = self._select_best_resolution(w, h)

        global_img = self.pad(image, (self.image_size, self.image_size))

        padded_img = self.pad(image, (best_w, best_h))

        local_tiles = []
        for i in range(0, best_h, self.image_size):
            for j in range(0, best_w, self.image_size):
                tile = padded_img.crop((j, i, j + self.image_size, i + self.image_size))
                local_tiles.append(tile)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )
        global_tensor = transform(global_img)
        local_tensors = [transform(t) for t in local_tiles]

        all_tiles = torch.stack([global_tensor] + local_tensors)

        return {
            "pixel_values": all_tiles,
            "num_width_tiles": best_w // self.image_size,
            "num_height_tiles": best_h // self.image_size,
        }


class DeepseekVLV2Processor(DeepseekVLProcessor, ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "num_image_tokens"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
    ):
        self.image_token = tokenizer.image_token
        super().__init__(image_processor, tokenizer, chat_template)

    def __call__(
        self,
        conversation,
        images=None,
        tokenize=True,
        return_tensors="pt",
        **kwargs,
    ):
        prompt = self.apply_chat_template(conversation, **kwargs)

        if not tokenize:
            return prompt

        batch_pixel_values = []
        batch_spatial_crops = []

        for img in images:
            out = self.image_processor.preprocess(img)
            batch_pixel_values.append(out["pixel_values"])
            batch_spatial_crops.append(
                [out["num_width_tiles"], out["num_height_tiles"]]
            )

        max_tiles = max(pv.shape[0] for pv in batch_pixel_values)
        padded_pixel_values = torch.zeros(len(batch_pixel_values), max_tiles, 3, 384, 384)
        for i, pv in enumerate(batch_pixel_values):
            padded_pixel_values[i, :pv.shape[0]] = pv

        images_spatial_crop = torch.zeros(len(batch_spatial_crops), max_tiles, 2, dtype=torch.long)
        for i, (w, h) in enumerate(batch_spatial_crops):
            images_spatial_crop[i, 0] = torch.tensor([w, h])

        expanded_prompt = prompt
        for w, h in batch_spatial_crops:
            num_tokens = (h * 14) * (w * 14 + 1) + 210 + 1
            expanded_prompt = expanded_prompt.replace(
                self.image_token, self.image_token * num_tokens, 1
            )

        enc = self.tokenizer(expanded_prompt, return_tensors=return_tensors)

        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        images_seq_mask = (enc["input_ids"] == image_token_id).to(torch.bool)

        labels = enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return BatchFeature(
            {
                "input_ids": enc["input_ids"],
                "labels" : labels,
                "attention_mask": enc["attention_mask"],
                "pixel_values": padded_pixel_values,
                "images_spatial_crop": images_spatial_crop,
                "images_seq_mask": images_seq_mask,
            }
        )

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        return ProcessorMixin.apply_chat_template(
            self, conversations, chat_template, **kwargs
        )
