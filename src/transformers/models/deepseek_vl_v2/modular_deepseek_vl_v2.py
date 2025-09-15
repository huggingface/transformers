from typing import Optional, Tuple, List
from einops import rearrange, repeat
import gc

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageOps

from ...configuration_utils import PretrainedConfig
from ..auto import AutoConfig, AutoModel, AutoModelForCausalLM, CONFIG_MAPPING
from ..deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config

from ..deepseek_vl.processing_deepseek_vl import DeepseekVLProcessor
from transformers.image_processing_utils import BatchFeature
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
        **kwargs
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.downsample_ratio = downsample_ratio

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
        "language_config": DeepseekV2Config,
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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        self.candidate_resolutions = candidate_resolutions
        self.n_embed = n_embed

        if language_config is None:
            language_config = {}
            logger.info("`text_config` is `None`. Initializing the `LlamaConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `SiglipVisionConfig` with default values.")

        if isinstance(language_config, dict):
            text_config["model_type"] = language_config.get("model_type", "deepseek_v2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "siglip_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        
        if isinstance(projector_config, dict):
            projector_config["model_type"] = projector_config.get("model_type", "mlp_projector")
            projector_config = CONFIG_MAPPING[projector_config["model_type"]](**projector_config)


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
        self.language_model = AutoModelForCausalLM(config.language_config)

        projector_config = config.projector_config
        self.projector = MlpProjector(projector_config)

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        embed_std = 1/torch.sqrt(torch.tensor(projector_config.n_embed, dtype=torch.float32))

        if self.tile_tag== "2D":
            self.image_newline = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
            self.view_seperator = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
        elif self.tile_tag == "1D":
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError("candidate_resolutions should not be empty")

            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, projector_config.n_embed) * embed_std)
            )
        else:
            raise ValueError(f"Unknown tile_tag: {self.tile_tag}. It should one of the following: '1D', '2D'.")

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
                num_width_tiles, num_height_tiles = images_spatial_crop[idx,jdx]

                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                batch_num_tiles[idx] += (1+num_width_tiles * num_height_tiles)

            total_tiles.append(batch_num_tiles[idx])

        total_tiles = torch.cat(total_tiles,dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)

        if total_tiles.shape[0] == 0:
            return self.language_model.get_input_embeddings()(input_ids)

        images_features = self.vision_model(
            total_tiles
        )

        images_embeds = self.projector(images_features)
        _, hw, n_dim = images_embeds.shape
        h = w = int(hw**0.5)

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        tile_index = 0
        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx,jdx]

                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles
                global_features = images_embeds[tile_index]

                local_features = images_embeds[tile_index+1 : tile_index+1 + num_tiles_in_image]

                tile_index += num_tiles_in_image + 1

                if self.tile_tag == "2D":
                    global_features = global_features.view(h,w,n_dim)
                    new_lines_in_global = repeat(self.image_newline,
                                                "d -> h 1 d", h=h
                                                 )
                    global_features = torch.cat([
                        global_features,new_lines_in_global
                    ],dim=1)
                    global_features = global_features.view(-1,n_dim)

                    local_features = rearrange(
                        local_features,
                        "(th tw) (h w) d -> (th h) (tw w) d",
                        th=num_height_tiles,
                        tw=num_width_tiles,         
                        h=h,
                        w=w,
                    )

                    new_lines_in_local = repeat(
                        self.image_newline,
                        "d -> (th h) 1 d",
                        th=num_height_tiles,
                        h=h
                    )

                    local_features = torch.cat([
                        local_features,new_lines_in_local
                    ],dim=1)

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat([
                            global_features,self.view_seperator[None,:], local_features
                        ],dim=0)
                    else:
                        global_local_features = torch.cat([
                            local_features,self.view_seperator[None,:], global_features
                        ],dim=0)
                else:
                    global_features = torch.cat(
                        [self.tile_indicators[0:1],global_features],dim=0
                    )

                    local_features = torch.cat(
                        [self.tile_indicators[1: num_tiles_in_image + 1].unsqueeze(1), local_features],dim=1
                    )

                    local_features = rearrange(
                        local_features,
                        "crop_num hw d -> (crop_num hw) d",
                    )

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat([
                            global_features, local_features
                        ],dim=0)
                    else:
                        global_local_features = torch.cat([
                            local_features, global_features
                        ],dim=0)

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch,dim=0)

                inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1),images_in_this_batch)

        return inputs_embeds

    @torch.no_grad()
    def incremental_prefilling(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        chunk_size: int = 1024,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                inputs_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

            del images
            del images_seq_mask
            del images_spatial_crop

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

            self._clear_cuda_cache()

        bs, seq_len, _ = inputs_embeds.shape
        past_key_values = None

        prefilling_len = seq_len - 1
        for i in range(0, prefilling_len, chunk_size):
            chunk_start = i
            chunk_end = min(i+chunk_size, prefilling_len)
            chunk_inputs_embeds = inputs_embeds[:, chunk_start: chunk_end]
            chunk_attention_mask = attention_mask[:, 0:chunk_end]

            if past_key_values is not None:
                position_ids = torch.arange(
                    chunk_start,
                    chunk_end,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                ).unsqueeze(0)
                past_key_values = self._move_past_key_values_to_gpu(past_key_values, inputs_embeds.device)
            else:
                position_ids = None

            with torch.no_grad():
                outputs = self.forward(
                    inputs_embeds=chunk_inputs_embeds,
                    attention_mask=chunk_attention_mask,
                    past_key_values=outputs.past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values,
                past_key_values = self._move_past_key_values_to_cpu(past_key_values)

                del outputs, position_ids
                self._clear_cuda_cache()

        prefilling_key_values = []
        for layer_past in past_key_values:
            prefilling_key_values.append(
                (
                    layer_past[0][:, :, 0:prefilling_len, ...].to(inputs_embeds.device),
                    layer_past[1][:, :, 0:prefilling_len, ...].to(inputs_embeds.device),
                )
            )

        return inputs_embeds, prefilling_key_values

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

        # print(inputs_embeds.shape)
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

    def _clear_cuda_cache(self):
        """clear CUDA memory cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _move_past_key_values_to_cpu(self, past_key_values):
        # print(f"past_key_values -> cpu")
        if past_key_values is None:
            return None
        return tuple(tuple(t.cpu() for t in layer) for layer in past_key_values)

    def _move_past_key_values_to_gpu(self, past_key_values, device="cuda:0"):
        # print(f"past_key_values -> gpu")
        if past_key_values is None:
            return None
        return tuple(tuple(t.to(device) for t in layer) for layer in past_key_values)

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
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = self.language.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model
        cache_position = model_inputs["cache_position"]
        if cache_position[0] == 0:
            model_inputs["images"] = images
            model_inputs["images_seq_mask"] = images_seq_mask
            model_inputs["images_spatial_crop"] = images_spatial_crop

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

class DeepseekVLV2Processor(DeepseekVLProcessor):
    """
    Transformers-compatible processor for Deepseek VL v2.
    Inherits from DeepseekVLProcessor but overrides the behavior to:
      - handle tiling and cropping (global + local views)
      - expand <image> placeholders into the right number of tokens
      - build `images_seq_mask`, `images_spatial_crop`, and `num_image_tokens`.
    """

    def __init__(
        self,
        *args,
        candidate_resolutions: List[Tuple[int, int]] = ((384, 384),),
        patch_size: int = 14,
        downsample_ratio: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.candidate_resolutions = candidate_resolutions
        self.image_size = candidate_resolutions[0][0]
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio

    def tokenize_with_images(self, text: str, images: List[Image.Image]):
        """
        Tokenize text with <image> placeholders expanded into correct number of image tokens.
        Also preprocess images into tiles + global view.
        """
        assert text.count(self.image_token) == len(
            images
        ), f"Mismatched <image> count ({text.count(self.image_token)}) vs {len(images)} images."

        (
            tokenized_str,
            images_list,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
        ) = ([], [], [], [], [])
        splits = text.split(self.image_token)

        for split_text, image in zip(splits, images):
            # text part
            token_ids = self.tokenizer.encode(split_text, add_special_tokens=False)
            tokenized_str.extend(token_ids)
            images_seq_mask.extend([False] * len(token_ids))

            # preprocess image: global + local views
            best_w, best_h = self.candidate_resolutions[
                0
            ]  # TODO: select best like official code
            global_view = ImageOps.pad(image, (self.image_size, self.image_size))
            images_list.append(self.image_processor(global_view)["pixel_values"])

            local_view = ImageOps.pad(image, (best_w, best_h))
            for i in range(0, best_h, self.image_size):
                for j in range(0, best_w, self.image_size):
                    crop = local_view.crop(
                        (j, i, j + self.image_size, i + self.image_size)
                    )
                    images_list.append(self.image_processor(crop)["pixel_values"])

            # track tiling shape
            num_w_tiles, num_h_tiles = (
                best_w // self.image_size,
                best_h // self.image_size,
            )
            images_spatial_crop.append([num_w_tiles, num_h_tiles])

            # expand <image> token → num_image_tokens
            h = w = (self.image_size // self.patch_size) // self.downsample_ratio
            image_token_ids = [
                self.tokenizer.convert_tokens_to_ids(self.image_token)
            ] * (h * (w + 1) + 1 + (num_h_tiles * h) * (num_w_tiles * w + 1))
            tokenized_str.extend(image_token_ids)
            images_seq_mask.extend([True] * len(image_token_ids))
            num_image_tokens.append(len(image_token_ids))

        # last text after final image
        last_split = splits[-1]
        if last_split:
            token_ids = self.tokenizer.encode(last_split, add_special_tokens=False)
            tokenized_str.extend(token_ids)
            images_seq_mask.extend([False] * len(token_ids))

        return (
            tokenized_str,
            images_list,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
        )

    def __call__(
        self, text=None, images=None, return_tensors="pt", **kwargs
    ) -> BatchFeature:
        """
        Override __call__ to return both text and image features.
        """
        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        if isinstance(text, str):
            text = [text]
        if images is None:
            images = [None] * len(text)

        (
            all_input_ids,
            all_attention_mask,
            all_images,
            all_masks,
            all_spatial,
            all_num_tokens,
        ) = ([], [], [], [], [], [])

        for t, im in zip(text, images):
            if im is None:
                enc = self.tokenizer(t, return_tensors=return_tensors)
                all_input_ids.append(enc["input_ids"].squeeze(0))
                all_attention_mask.append(enc["attention_mask"].squeeze(0))
                all_images.append(torch.zeros(1, 3, self.image_size, self.image_size))
                all_masks.append(
                    torch.zeros_like(enc["input_ids"], dtype=torch.bool).squeeze(0)
                )
                all_spatial.append(torch.zeros((1, 2), dtype=torch.long))
                all_num_tokens.append([0])
            else:
                ids, ims, mask, spatial, num_tokens = self.tokenize_with_images(t, [im])
                all_input_ids.append(torch.tensor(ids))
                all_attention_mask.append(torch.ones(len(ids), dtype=torch.long))
                all_images.append(torch.stack(ims))
                all_masks.append(torch.tensor(mask, dtype=torch.bool))
                all_spatial.append(torch.tensor(spatial, dtype=torch.long))
                all_num_tokens.append(num_tokens)

        # pad sequences
        input_ids = pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad_sequence(
            all_attention_mask, batch_first=True, padding_value=0
        )
        images_seq_mask = pad_sequence(all_masks, batch_first=True, padding_value=0)

        return BatchFeature(
            data=dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=torch.nested.to_padded_tensor(
                    torch.nested.nested_tensor(all_images), 0.0
                ),
                images_seq_mask=images_seq_mask,
                images_spatial_crop=all_spatial,
                num_image_tokens=all_num_tokens,
            )
        )
