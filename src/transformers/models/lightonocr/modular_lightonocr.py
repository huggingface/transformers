from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn

from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_utils import ImageInput
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_vision_available
from ..pixtral.configuration_pixtral import PixtralVisionConfig
from ..pixtral.image_processing_pixtral import get_resize_output_image_size
from ..pixtral.modeling_pixtral import (
    PixtralVisionModel,
)
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import (
    Qwen3Model,
    Qwen3RMSNorm,
)


if is_vision_available():
    from ..pixtral.image_processing_pixtral import get_resize_output_image_size


class LightOnOCRTextConfig(Qwen3Config):
    model_type = "lightonocr_text"
    # Remove unused attributes inherited from Qwen3Config
    max_window_layers = None
    use_sliding_window = None


class LightOnOCRVisionConfig(PixtralVisionConfig):
    model_type = "lightonocr_vision"
    pass


class LightOnOCRConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LightOnOCRForConditionalGeneration`]. It is used to instantiate a
    LightOnOCR model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a configuration with the defaults will yield
    a similar configuration to that of the LightOnOCR [lightonocr-hf/lightonocr-9b](https://huggingface.co/lightonocr-hf/lightonocr-9b) architecture.

    Args:
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size of spatial merging for image patches.
        vision_config (`dict` or `LightOnOCRVisionConfig`, *optional*):
            Custom vision configuration or dictionary with vision configuration values.
        text_config (`dict` or `LightOnOCRTextConfig`, *optional*):
            Custom text configuration or dictionary with text configuration values.

    Example:

    ```python
    >>> from transformers import LightOnOCRConfig, LightOnOCRForConditionalGeneration

    >>> # Initializing a LightOnOCR configuration
    >>> configuration = LightOnOCRConfig()

    >>> # Initializing a model from the configuration
    >>> model = LightOnOCRForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "lightonocr"
    sub_configs = {"text_config": LightOnOCRTextConfig, "vision_config": LightOnOCRVisionConfig}

    def __init__(
        self,
        spatial_merge_size: int = 2,
        image_token_index: int = 151655,
        tie_word_embeddings=True,
        vision_config: Optional[dict[str, Any]] = None,
        text_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        self.spatial_merge_size = spatial_merge_size
        self.image_token_id = image_token_index

        if vision_config is None:
            self.vision_config = LightOnOCRVisionConfig(
                attention_dropout=0,
                head_dim=64,
                hidden_act="silu",
                hidden_size=1024,
                image_size=1540,
                initializer_range=0.02,
                intermediate_size=4096,
                model_type="pixtral",
                num_attention_heads=16,
                num_channels=3,
                num_hidden_layers=24,
                patch_size=14,
                rope_theta=10000,
            )
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        else:
            self.vision_config = LightOnOCRVisionConfig(**vision_config)

        if text_config is None:
            self.text_config = LightOnOCRTextConfig(
                architectures=["Qwen3ForCausalLM"],
                attention_dropout=0,
                head_dim=128,
                hidden_act="silu",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=3072,
                max_position_embeddings=40960,
                model_type="qwen3",
                num_attention_heads=16,
                num_hidden_layers=28,
                num_key_value_heads=8,
                rms_norm_eps=1e-6,
                rope_theta=1000000,
                sliding_window=None,
                use_cache=True,
                use_sliding_window=False,
                vocab_size=151936,
            )
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        else:
            self.text_config = LightOnOCRTextConfig(**text_config)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class LightOnOCRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "patch_size": 14,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


class LightOnOCRProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        chat_template=None,
        image_token="<|image_pad|>",
        image_break_token="<|vision_pad|>",
        image_end_token="<|vision_end|>",
        **kwargs,
    ):
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        # Calculate effective patch size for image processing
        self.effective_patch_size = patch_size * spatial_merge_size
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_break_token = image_break_token
        self.image_end_token = image_end_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_break_token_id = tokenizer.convert_tokens_to_ids(self.image_break_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)
        self.image_ids = [self.image_token_id, self.image_break_token_id, self.image_end_token_id]

        # Set the default patch_size for images_kwargs
        LightOnOCRProcessorKwargs._defaults["images_kwargs"]["patch_size"] = self.effective_patch_size

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        **kwargs,
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You must provide either text or images")
        output_kwargs = self._merge_kwargs(
            LightOnOCRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        # Expand image token if image is present
        if image_inputs.get("pixel_values") is not None:
            height, width = image_inputs["image_sizes"][0]
            num_height_tokens = height // self.effective_patch_size
            num_width_tokens = width // self.effective_patch_size
            num_patches = num_height_tokens * num_width_tokens

            # Replace single image token with repeated tokens
            expanded_tokens = self.image_token * num_patches
            prompt_strings = [sample.replace(self.image_token, expanded_tokens) for sample in text]
        else:
            prompt_strings = text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[np.isin(array_ids, self.image_ids)] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = LightOnOCRProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            size = images_kwargs.get("size", None) or self.image_processor.size

            num_image_tokens = []
            for height, width in image_sizes:
                resized_height, resized_width = get_resize_output_image_size(
                    np.zeros((height, width, 3)),
                    size=(size["longest_edge"], size["longest_edge"]),
                    patch_size=(self.effective_patch_size, self.effective_patch_size),
                )
                num_height_tokens = resized_height // self.effective_patch_size
                num_width_tokens = resized_width // self.effective_patch_size
                num_image_tokens.append((num_width_tokens + 1) * num_height_tokens)

            num_image_patches = [1] * len(image_sizes)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)


# Text model RMSNorm defined early for use in MultiModalProjector
class LightOnOCRTextRMSNorm(Qwen3RMSNorm):
    pass


class LightOnOCRPatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches.
    """

    def __init__(self, config: LightOnOCRConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_size = config.vision_config.patch_size

        self.merging_layer = nn.Linear(self.hidden_size * self.spatial_merge_size**2, self.hidden_size, bias=False)

    def forward(self, image_features: torch.Tensor, image_sizes: list[tuple[int, int]]) -> torch.Tensor:
        image_sizes_in_patches = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes_in_patches]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(image_features.split(tokens_per_image)):
            # reshape image_tokens into a 2D grid
            h, w = image_sizes_in_patches[image_index]
            # shape [num_patches, d] -> [1, d, h, w]
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            # shape [1, d, h, w] -> [h // sms * w // sms, d * sms**2]
            # sms = spatial_merge_size
            # note(staghado): when h or w is not divisible by sms, the last row/column will be ignored??
            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )
            # shape [h // sms * w // sms, d * sms**2] -> [h // sms * w // sms, d * sms**2]
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        image_features = self.merging_layer(image_features)
        return image_features


class LightOnOCRVisionProjector(nn.Module):
    def __init__(self, config: LightOnOCRConfig):
        super().__init__()
        self.config = config

        self.norm = LightOnOCRTextRMSNorm(config.vision_config.hidden_size, eps=1e-6)
        self.patch_merger = LightOnOCRPatchMerger(config)
        self.act = nn.GELU()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=False,
        )
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)

    def forward(self, image_features: torch.Tensor, image_sizes):
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LightOnOCRPreTrainedModel(PreTrainedModel):
    config_class = LightOnOCRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LightOnOCRVisionProjector", "LightOnOCRPatchMerger"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        # Determine which component this module belongs to by checking the module path
        module_name = None
        for name, mod in self.named_modules():
            if mod is module:
                module_name = name
                break

        # Use appropriate initializer range based on module path
        if module_name and module_name.startswith("vision_encoder"):
            std = (
                self.config.vision_config.initializer_range
                if hasattr(self.config.vision_config, "initializer_range")
                else 0.02
            )
        elif module_name and module_name.startswith("language_model"):
            std = (
                self.config.text_config.initializer_range
                if hasattr(self.config.text_config, "initializer_range")
                else 0.02
            )
        else:
            # For projector and other components, use language model's initializer range
            std = 0.02

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Text model components - explicitly renamed from Qwen3 (LightOnOCRTextRMSNorm already defined above)
class LightOnOCRTextPreTrainedModel(PreTrainedModel):
    config_class = LightOnOCRTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LightOnOCRTextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True


@auto_docstring(
    custom_intro="""
    The language model of LightOnOCR, based on Qwen3 architecture.
    """
)
class LightOnOCRTextModel(Qwen3Model):
    config_class = LightOnOCRTextConfig

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


# Vision model components - explicitly renamed from Pixtral
class LightOnOCRVisionPreTrainedModel(PreTrainedModel):
    config_class = LightOnOCRVisionConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_attention_backend = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _no_split_modules = ["LightOnOCRVisionAttentionLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        else:
            module.weight.data.fill_(1.0)


@auto_docstring(
    custom_intro="""
    The vision encoder of LightOnOCR, based on Pixtral vision architecture.
    """
)
class LightOnOCRVisionModel(PixtralVisionModel):
    config_class = LightOnOCRVisionConfig


class LightOnOCRModel(LightOnOCRPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: LightOnOCRConfig

    def __init__(self, config: LightOnOCRConfig):
        super().__init__(config)

        self.vision_encoder = LightOnOCRVisionModel(config.vision_config)

        self.vision_projection = LightOnOCRVisionProjector(config)

        self.language_model = LightOnOCRTextModel(config.text_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values: torch.Tensor, image_sizes: list[tuple[int, int]]):
        visual_features = self.vision_encoder(pixel_values, image_sizes=image_sizes).last_hidden_state

        image_features = self.vision_projection(visual_features.squeeze(0), image_sizes)

        return image_features

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[list[tuple[int, int]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")

            # Get text embeddings
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # If pixel_values is provided, process vision encoder
        if pixel_values is not None:
            # Process image through the vision encoder
            visual_features = self.vision_encoder(pixel_values, image_sizes=image_sizes).last_hidden_state
            projected_visual = self.vision_projection(visual_features.squeeze(0), image_sizes)

            # Convert to same dtype
            projected_visual = projected_visual.to(inputs_embeds.dtype)

            # Get mask for image tokens using get_placeholder_mask
            image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, projected_visual)

            # Replace image tokens with visual embeddings using masked_scatter
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, projected_visual)

        # Get language model outputs
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if not return_dict:
            return outputs

        # Return BaseModelOutputWithPast with all relevant information
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LightOnOCRForConditionalGeneration(LightOnOCRPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    config_class = LightOnOCRConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LightOnOCRConfig):
        super().__init__(config)
        self.model = LightOnOCRModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model.language_model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[list[tuple[int, int]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        logits: torch.Tensor = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

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
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = kwargs.get("image_sizes")

        return model_inputs

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_encoder(self):
        return self.model.vision_encoder


__all__ = [
    "LightOnOCRPreTrainedModel",
    "LightOnOCRTextModel",
    "LightOnOCRTextPreTrainedModel",
    "LightOnOCRVisionModel",
    "LightOnOCRVisionPreTrainedModel",
    "LightOnOCRForConditionalGeneration",
    "LightOnOCRModel",
    "LightOnOCRConfig",
    "LightOnOCRProcessor",
]
