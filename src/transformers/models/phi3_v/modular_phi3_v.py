# coding=utf-8
# Copyright 2025 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""PyTorch Phi-3-V model."""

from typing import Optional, Union

import torch
import torch.nn as nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..deepseek_vl.modeling_deepseek_vl import DeepseekVLForConditionalGeneration
from ..janus.modeling_janus import (
    JanusBaseModelOutputWithPast,
    JanusCausalLMOutputWithPast,
)
from ..llama4.modeling_llama4 import Llama4VisionMLP


logger = logging.get_logger(__name__)


class Phi3VConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Phi3VModel`]. It is used to instantiate an
    Phi3 Vision model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Phi-3.5 vision model.

    e.g. [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Phi3Config`):
            The config object or dictionary of the text backbone.
        image_token_id (`int`, *optional*, defaults to 32044):
            Token index of a placeholder image token.

    Example:

    ```python
    >>> from transformers import Phi3VForConditionalGeneration, Phi3VConfig, Phi3Config, CLIPVisionConfig

    >>> # Initializing vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing text config
    >>> text_config = Phi3Config()

    >>> # Initializing config
    >>> configuration = Phi3VConfig(vision_config=vision_config, text_config=text_config)

    >>> # Initializing a model from the Phi-3 style configuration
    >>> model = Phi3VForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "phi3_v"

    sub_configs = {"vision_config": AutoConfig, "text_config": AutoConfig}

    def __init__(self, vision_config=None, text_config=None, image_token_id=32044, **kwargs):
        if text_config is None:
            logger.info("`text_config` is None. Initializing with default values")
            self.text_config = CONFIG_MAPPING["phi3"]()
        elif isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "phi3")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        else:
            raise ValueError(
                f"Invalid type for `text_config`. Must be either `dict` or `LlamaConfig`."
                f" Type found: {type(text_config)}"
            )

        if vision_config is None:
            logger.info("`vision_config` is None. Initializing with default CLIPVisionConfig values")
            self.vision_config = CONFIG_MAPPING["clip_vision_model"]()
        elif isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "clip_vision_model")
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        else:
            raise ValueError(
                f"Invalid type for `vision_config`. Must be either `dict` or `CLIPVisionConfig`."
                f" Type found: {type(vision_config)}"
            )

        self.image_token_id = image_token_id
        super().__init__(**kwargs)


class Phi3VBaseModelOutputWithPast(JanusBaseModelOutputWithPast):
    pass


class Phi3VCausalLMOutputWithPast(JanusCausalLMOutputWithPast):
    pass


@auto_docstring
class Phi3VPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models. The model is only intended for inference and doesn't support finetuning.
    """

    config: Phi3VConfig
    base_model_prefix = "model"
    input_modalities = ["image", "text"]
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_param_buffer_assignment = False

    def _init_weights(self, module):
        std = self.config.get_text_config().initializer_range
        super()._init_weights(module)
        if hasattr(module, "sub_newline"):
            if isinstance(module.sub_newline, nn.Parameter):
                init.normal_(module.sub_newline, mean=0.0, std=std)
        if hasattr(module, "glb_newline"):
            if isinstance(module.glb_newline, nn.Parameter):
                init.normal_(module.glb_newline, mean=0.0, std=std)


class Phi3VImageProjection(Llama4VisionMLP, nn.Module):
    def __init__(self, config):
        nn.Module.__init__()
        self.config = config
        in_dim, out_dim = config.get_text_config().hidden_size, config.vision_config.intermediate_size
        self.fc1 = nn.Linear(out_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)
        self.activation_fn = nn.GELU()


class Phi3VModel(Phi3VPreTrainedModel):
    def __init__(self, config: Phi3VConfig):
        super().__init__(config)
        self.config = config
        self.image_dim_out = config.vision_config.hidden_size

        self.vision_model = AutoModel.from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.image_projection = Phi3VImageProjection(config)

        self.glb_newline = nn.Parameter(torch.zeros([1, 1, self.image_dim_out * 4]))
        self.sub_newline = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out * 4]))

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        """Reshape high-dimensional patches by merging 2x2 patches."""
        batch_size, num_patches, in_channels = image_features.shape
        height_patches = int(num_patches**0.5)

        num_images = batch_size // (h_crop * w_crop)

        # Reshape into patch grid
        image_features_grid = image_features.reshape(batch_size, height_patches, height_patches, in_channels)

        # Split each dimension into (H/2, 2) so we can merge 2×2 blocks
        image_features_grid = image_features_grid.reshape(
            batch_size, height_patches // 2, 2, height_patches // 2, 2, in_channels
        )

        image_features_grid = image_features_grid.permute(0, 1, 3, 2, 4, 5)
        merged_features = image_features_grid.reshape(batch_size, -1, 4 * in_channels)

        merged_features = merged_features.reshape(
            num_images, h_crop, w_crop, height_patches // 2, height_patches // 2, -1
        )

        merged_features = merged_features.permute(0, 1, 3, 2, 4, 5)
        output = merged_features.reshape(
            num_images, h_crop * (height_patches // 2), w_crop * (height_patches // 2), 4 * in_channels
        )

        return output

    def add_newline_embeds(self, image_features):
        """Add the newline token embeds to the image feature patches"""
        num_images, h, w, hid_dim = image_features.shape
        newline_embeddings = self.sub_newline.expand(num_images, h, -1, -1)
        image_features_newline = torch.cat([image_features, newline_embeddings], dim=2)
        image_features_newline = image_features_newline.reshape(num_images, -1, hid_dim)
        return image_features_newline

    def transform_image_embeds(self, hidden_states: torch.Tensor, image_sizes) -> torch.Tensor:
        """Process the output of vision model to obtain image embeddings suitable for multimodal model input."""
        global_image_features = hidden_states[:, 0]  # (num_images, 24*24, 1024)
        global_image_features_hd = self.reshape_hd_patches_2x2merge(global_image_features, 1, 1)
        global_image_features_hd_newline = self.add_newline_embeds(global_image_features_hd)

        all_image_embeddings = []
        # need a for loop to process each image because of different image sizes
        # (patch arrangement is different for each image)
        for i, img_size in enumerate(image_sizes):
            h, w = img_size
            h_crop = h // self.config.vision_config.image_size
            w_crop = w // self.config.vision_config.image_size
            num_crops = h_crop * w_crop

            # NOTE: real num_crops is padded (num_crops, 24*24, 1024)
            sub_image_features = hidden_states[i, 1 : 1 + num_crops]
            sub_image_features_hd = self.reshape_hd_patches_2x2merge(sub_image_features, h_crop, w_crop)
            sub_image_features_hd_newline = self.add_newline_embeds(sub_image_features_hd)

            # [sub features, separator, global features]
            all_image_embeddings.extend(
                [
                    sub_image_features_hd_newline.squeeze(0),
                    self.glb_newline.squeeze(0),
                    global_image_features_hd_newline[i],
                ]
            )

        image_embeds = torch.cat(all_image_embeddings, dim=0)
        image_embeds_proj = self.image_projection(image_embeds)

        return image_embeds_proj

    def get_image_features(self, pixel_values: torch.Tensor, image_sizes, num_images, num_crops):
        # Process image using CLIP model.
        vision_outputs = self.vision_model(pixel_values, output_hidden_states=True)

        # Extract the hidden states from the second last layer.
        hidden_state = vision_outputs.hidden_states[-2][:, 1:]
        hidden_state = hidden_state.reshape(num_images, num_crops, -1, self.image_dim_out)

        # Transform the image features to text embedding space.
        image_features = self.transform_image_embeds(hidden_state, image_sizes)
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
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            n_image_features = image_features.shape[0] * image_features.shape[1]
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_sizes: torch.Tensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            num_images, num_crops, c, h, w = pixel_values.shape
            pixel_values = pixel_values.flatten(0, 1)
            image_features = self.get_image_features(pixel_values, image_sizes, num_images, num_crops)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_attention_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(image_attention_mask, image_features)

        lm_output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        return Phi3VBaseModelOutputWithPast(
            last_hidden_state=lm_output.last_hidden_state,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class Phi3VForConditionalGeneration(DeepseekVLForConditionalGeneration):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_sizes: torch.Tensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_sizes=image_sizes,
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

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Phi3VCausalLMOutputWithPast(
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
        pixel_values=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        logits_to_keep=None,
        image_sizes=None,
        **kwargs,
    ):
        # Overwritten -- extra custom processing
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            image_sizes=image_sizes,
            **kwargs,
        )

        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


__all__ = ["Phi3VConfig", "Phi3VModel", "Phi3VPreTrainedModel", "Phi3VForConditionalGeneration"]
