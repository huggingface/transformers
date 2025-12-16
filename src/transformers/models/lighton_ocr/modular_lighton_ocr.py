from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn

from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_utils import ImageInput
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring
from ...utils.generic import TransformersKwargs, check_model_inputs
from ..pixtral.configuration_pixtral import PixtralVisionConfig
from ..pixtral.image_processing_pixtral import get_resize_output_image_size
from ..pixtral.modeling_pixtral import (
    PixtralAttention,
    PixtralRMSNorm,
    PixtralVisionModel,
    apply_rotary_pos_emb,
)
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import (
    Qwen3Model,
)


class LightOnOcrVisionConfig(PixtralVisionConfig):
    model_type = "lighton_ocr_vision"


class LightOnOcrTextConfig(Qwen3Config):
    model_type = "lighton_ocr_text"


class LightOnOcrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LightOnOcrForConditionalGeneration`]. It is used to instantiate a
    LightOnOcr model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a configuration with the defaults will yield
    a similar configuration to that of the LightOnOcr [lightonocr-hf/lightonocr-9b](https://huggingface.co/lightonocr-hf/lightonocr-9b) architecture.

    Args:
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size of spatial merging for image patches.
        image_token_id (`int`, *optional*, defaults to 151655):
            The id of the image token in the vocabulary.
        vision_config (`dict` or `LightOnOcrVisionConfig`, *optional*):
            Custom vision configuration or dictionary with vision configuration values.
        text_config (`dict` or `LightOnOcrTextConfig`, *optional*):
            Custom text configuration or dictionary with text configuration values.

    Example:

    ```python
    >>> from transformers import LightOnOcrConfig, LightOnOcrForConditionalGeneration

    >>> # Initializing a LightOnOcr configuration
    >>> configuration = LightOnOcrConfig()

    >>> # Initializing a model from the configuration
    >>> model = LightOnOcrForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "lighton_ocr"
    sub_configs = {"text_config": LightOnOcrTextConfig, "vision_config": LightOnOcrVisionConfig}

    def __init__(
        self,
        spatial_merge_size: int = 2,
        image_token_id: int = 151655,
        vision_config: Optional[dict[str, Any]] = None,
        text_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        self.spatial_merge_size = spatial_merge_size
        self.image_token_id = image_token_id

        if vision_config is None:
            self.vision_config = LightOnOcrVisionConfig(
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
            self.vision_config = LightOnOcrVisionConfig(**vision_config)

        if text_config is None:
            self.text_config = LightOnOcrTextConfig(
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
            self.text_config = LightOnOcrTextConfig(**text_config)

        super().__init__(**kwargs)


class LightOnOcrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


class LightOnOcrProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        chat_template=None,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        # Calculate effective patch size for image processing
        self.effective_patch_size = patch_size * spatial_merge_size

        # Get special tokens from tokenizer attributes
        # These should be set on the tokenizer before creating the processor
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.image_break_token = getattr(tokenizer, "image_break_token", "<|vision_pad|>")
        self.image_end_token = getattr(tokenizer, "image_end_token", "<|vision_end|>")

        # Get token IDs by converting from token strings
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_break_token_id = tokenizer.convert_tokens_to_ids(self.image_break_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)

        self.image_ids = [self.image_token_id, self.image_break_token_id, self.image_end_token_id]

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        **kwargs: Unpack[LightOnOcrProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You must provide either text or images")
        output_kwargs = self._merge_kwargs(
            LightOnOcrProcessorKwargs,
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

        if image_inputs.get("pixel_values") is not None:
            image_sizes_iter = iter(image_inputs["image_sizes"])
            prompt_strings = []

            for sample in text:
                replace_strings = []

                while self.image_token in sample:
                    image_height, image_width = next(image_sizes_iter)
                    num_height_tokens = image_height // self.effective_patch_size
                    num_width_tokens = image_width // self.effective_patch_size
                    num_patches = num_height_tokens * num_width_tokens

                    replace_str = self.image_token * num_patches
                    replace_strings.append(replace_str)

                    sample = sample.replace(self.image_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    replace_str = replace_strings.pop(0)
                    sample = sample.replace("<placeholder>", replace_str, 1)

                prompt_strings.append(sample)
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
            images_kwargs = LightOnOcrProcessorKwargs._defaults.get("images_kwargs", {})
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
                num_image_tokens.append(num_width_tokens * num_height_tokens)

            num_image_patches = [1] * len(image_sizes)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)


class LightOnOcrRMSNorm(PixtralRMSNorm):
    pass


class LightOnOcrPatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches.
    """

    def __init__(self, config: LightOnOcrConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_size = config.vision_config.patch_size

        self.merging_layer = nn.Linear(self.hidden_size * self.spatial_merge_size**2, self.hidden_size, bias=False)

    def forward(self, image_features: torch.Tensor, image_sizes: Union[torch.Tensor, list]) -> torch.Tensor:
        image_sizes_in_patches = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
        ]

        tokens_per_image = [patch_height * patch_width for patch_height, patch_width in image_sizes_in_patches]
        hidden_dim = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(image_features.split(tokens_per_image)):
            # reshape image_tokens into a 2D grid
            patch_height, patch_width = image_sizes_in_patches[image_index]
            # shape [num_patches, hidden_dim] -> [1, hidden_dim, patch_height, patch_width]
            image_grid = image_tokens.view(patch_height, patch_width, hidden_dim).permute(2, 0, 1).unsqueeze(0)
            # shape [1, hidden_dim, patch_height, patch_width] -> [patch_height // sms * patch_width // sms, hidden_dim * sms**2]
            # sms = spatial_merge_size
            # Note: patch_height and patch_width are guaranteed to be divisible by sms because the image processor
            # resizes images to multiples of effective_patch_size (patch_size * spatial_merge_size)
            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )
            # shape [patch_height // sms * patch_width // sms, hidden_dim * sms**2] -> [patch_height // sms * patch_width // sms, hidden_dim * sms**2]
            grid = grid.view(hidden_dim * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        image_features = self.merging_layer(image_features)
        return image_features


class LightOnOcrVisionProjector(nn.Module):
    def __init__(self, config: LightOnOcrConfig):
        super().__init__()
        self.config = config

        self.norm = LightOnOcrRMSNorm(config.vision_config.hidden_size, eps=1e-6)
        self.patch_merger = LightOnOcrPatchMerger(config)
        self.act = nn.GELU()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=False,
        )
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)

    def forward(self, image_features: torch.Tensor, image_sizes: Union[torch.Tensor, list]):
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LightOnOcrPreTrainedModel(PreTrainedModel):
    config_class = LightOnOcrConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LightOnOcrVisionProjector", "LightOnOcrPatchMerger", "LightOnOcrVisionAttentionLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_flex_attn = True
    _supports_attention_backend = True


# Copied from transformers.models.siglip.modeling_siglip.eager_attention_forward
def vision_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LightOnOcrAttention(PixtralAttention):
    """
    Multi-headed attention using vision_eager_attention_forward to avoid
    naming collision with Qwen3's eager_attention_forward (which has GQA support).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=0)

        attention_interface: Callable = vision_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Since we use packing, if flash_attention_2 is selected we rely on position_ids
        if self.config._attn_implementation == "flash_attention_2":
            kwargs["position_ids"] = kwargs["position_ids"].to(hidden_states.device, non_blocking=True)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


@auto_docstring(
    custom_intro="""
    The vision encoder of LightOnOcr, based on Pixtral vision architecture.
    """
)
class LightOnOcrVisionModel(PixtralVisionModel):
    config_class = LightOnOcrVisionConfig


@auto_docstring(
    custom_intro="""
    The language model of LightOnOcr, based on Qwen3 architecture.
    """
)
class LightOnOcrTextModel(Qwen3Model):
    config_class = LightOnOcrTextConfig

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class LightOnOcrModel(LightOnOcrPreTrainedModel):
    base_model_prefix = "model"
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: LightOnOcrConfig

    def __init__(self, config: LightOnOcrConfig):
        super().__init__(config)

        self.vision_encoder = LightOnOcrVisionModel._from_config(config.vision_config)

        self.vision_projection = LightOnOcrVisionProjector(config)

        self.language_model = LightOnOcrTextModel._from_config(config.text_config)

        self.post_init()

    @property
    def vision_model(self):
        """Alias for vision_encoder to match standard composite model naming."""
        return self.vision_encoder

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values: torch.Tensor, image_sizes: Union[torch.Tensor, list]):
        """
        Obtains image features from the vision encoder and projection.

        Args:
            pixel_values: Image tensors
            image_sizes: Tensor or list of (height, width) pairs for each image

        Returns:
            List of image feature tensors, one per image
        """
        visual_features = self.vision_encoder(pixel_values, image_sizes=image_sizes).last_hidden_state

        image_features = self.vision_projection(visual_features.squeeze(0), image_sizes)

        # Split features per image based on the effective patch size
        downsample_ratio = self.config.vision_config.patch_size * self.config.spatial_merge_size
        split_sizes = [(height // downsample_ratio) * (width // downsample_ratio) for height, width in image_sizes]
        image_features = torch.split(image_features, split_sizes)

        return image_features

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

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[Union[torch.Tensor, list]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            # Note: image_sizes is automatically expanded by the generation framework during beam search
            image_features_list = self.get_image_features(pixel_values, image_sizes)
            image_features = torch.cat(image_features_list, dim=0)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )

        return outputs


class LightOnOcrForConditionalGeneration(LightOnOcrPreTrainedModel, GenerationMixin):
    config_class = LightOnOcrConfig
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: LightOnOcrConfig):
        super().__init__(config)
        self.model = LightOnOcrModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[Union[torch.Tensor, list]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
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
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size)

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


__all__ = [
    "LightOnOcrPreTrainedModel",
    "LightOnOcrVisionModel",
    "LightOnOcrVisionPreTrainedModel",  # noqa: F822
    "LightOnOcrTextModel",
    "LightOnOcrTextPreTrainedModel",  # noqa: F822
    "LightOnOcrForConditionalGeneration",
    "LightOnOcrModel",
    "LightOnOcrConfig",
    "LightOnOcrTextConfig",
    "LightOnOcrVisionConfig",
    "LightOnOcrProcessor",
]
