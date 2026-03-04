# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
"""PI0 model: PaliGemma + Action Expert with flow matching for robot action prediction."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, can_return_tuple, logging
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel


logger = logging.get_logger(__name__)


class PI0ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "max_length": 48,
        },
        "images_kwargs": {
            "data_format": "channels_first",
        },
    }


@auto_docstring
class PI0Processor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")
        self.image_seq_length = image_processor.image_seq_length
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[PI0ProcessorKwargs],
    ) -> BatchFeature:
        """
        PI0 processor tokenizes language and supports multi-camera images per sample.
        Unlike `PaliGemmaProcessor`, it does not inject `<image>` placeholder tokens because PI0 concatenates image
        embeddings and language embeddings directly in the model.
        """
        output_kwargs = self._merge_kwargs(
            PI0ProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )

        if images is None:
            raise ValueError("`images` are expected as arguments to a `PI0Processor` instance.")
        if text is None:
            logger.warning_once("You are using PI0 without a text prefix. The processor will use an empty prompt.")
            text = ""

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, (list, tuple)):
            raise ValueError("`text` must be a string or a list of strings.")
        elif any(not isinstance(sample, str) for sample in text):
            raise ValueError("`text` must be a string or a list of strings.")

        text = [sample if sample.endswith("\n") else f"{sample}\n" for sample in text]

        if is_valid_image(images):
            batched_images = [[images]]
        elif isinstance(images, (list, tuple)) and len(images) > 0 and is_valid_image(images[0]):
            batched_images = [[image] for image in images]
        elif (
            isinstance(images, (list, tuple))
            and len(images) > 0
            and isinstance(images[0], (list, tuple))
            and len(images[0]) > 0
            and is_valid_image(images[0][0])
        ):
            batched_images = [list(sample_images) for sample_images in images]
        else:
            raise ValueError("`images` must be an image, a list of images, or a list of list of images.")

        if len(batched_images) != len(text):
            raise ValueError(
                f"Received {len(batched_images)} image samples for {len(text)} prompts. "
                "Each prompt should be associated with one sample (with one or more camera images)."
            )

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        tokenized = self.tokenizer(text, return_token_type_ids=True, **output_kwargs["text_kwargs"])

        max_num_cameras = max(len(sample_images) for sample_images in batched_images)
        image_masks = []
        padded_pixel_values = []

        for sample_images in batched_images:
            image_kwargs = dict(output_kwargs["images_kwargs"])
            image_kwargs.pop("return_tensors", None)
            processed = self.image_processor(sample_images, return_tensors="pt", **image_kwargs)["pixel_values"]
            num_cameras = processed.shape[0]

            sample_mask = torch.zeros(max_num_cameras, dtype=torch.bool)
            sample_mask[:num_cameras] = True
            image_masks.append(sample_mask)

            if num_cameras < max_num_cameras:
                pad = torch.zeros(
                    max_num_cameras - num_cameras,
                    *processed.shape[1:],
                    dtype=processed.dtype,
                )
                processed = torch.cat([processed, pad], dim=0)

            padded_pixel_values.append(processed)

        pixel_values = torch.stack(padded_pixel_values, dim=0)
        image_masks = torch.stack(image_masks, dim=0)

        return_data = {**tokenized, "pixel_values": pixel_values, "image_masks": image_masks}
        return BatchFeature(data=return_data, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            num_image_tokens = [self.image_seq_length] * len(image_sizes)
            num_image_patches = [1] * len(image_sizes)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        tokenizer_input_names = list(self.tokenizer.model_input_names)
        if "token_type_ids" not in tokenizer_input_names:
            tokenizer_input_names.append("token_type_ids")
        image_input_names = list(self.image_processor.model_input_names)
        names = tokenizer_input_names + image_input_names
        if "image_masks" not in names:
            names.append("image_masks")
        return names


class PI0Config(PreTrainedConfig):
    r"""
    Configuration class for PI0.

    PI0 is a robot action prediction model that combines a PaliGemma VLM backbone
    with an action expert Gemma model. It uses flow matching for continuous action generation.

    This model inherits from [`PaliGemmaConfig`]. See the superclass documentation for more details.
    Example checkpoint: [lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base).

    Args:
        vlm_config (`dict`, *optional*):
            Configuration for the vlm backbone (PaliGemmaModel).
        dit_config (`dict`, *optional*):
            Configuration for the DiT backbone. Defaults to a Gemma 300M variant.
        chunk_size (`int`, *optional*, defaults to 50):
            Number of action steps to predict per chunk.
        max_state_dim (`int`, *optional*, defaults to 32):
            Maximum state vector dimension (shorter vectors are zero-padded).
        max_action_dim (`int`, *optional*, defaults to 32):
            Maximum action vector dimension (shorter vectors are zero-padded).
        num_inference_steps (`int`, *optional*, defaults to 10):
            Number of denoising steps during inference.
        time_sampling_beta_alpha (`float`, *optional*, defaults to 1.5):
            Alpha parameter for Beta distribution used to sample diffusion time during training.
        time_sampling_beta_beta (`float`, *optional*, defaults to 1.0):
            Beta parameter for Beta distribution used to sample diffusion time during training.
        time_sampling_scale (`float`, *optional*, defaults to 0.999):
            Scale factor for sampled time values.
        time_sampling_offset (`float`, *optional*, defaults to 0.001):
            Offset added to sampled time values.
        min_period (`float`, *optional*, defaults to 0.004):
            Minimum period for sinusoidal time embedding.
        max_period (`float`, *optional*, defaults to 4.0):
            Maximum period for sinusoidal time embedding.

    Example:
    ```python
    >>> from transformers import PI0ForConditionalGeneration, PI0Config

    >>> config = PI0Config()
    >>> model = PI0ForConditionalGeneration(config)
    ```
    """

    model_type = "pi0"
    sub_configs = {"vlm_config": AutoConfig, "dit_config": AutoConfig}

    def __init__(
        self,
        vlm_config=None,
        dit_config=None,
        chunk_size=50,
        max_state_dim=32,
        max_action_dim=32,
        num_inference_steps=10,
        time_sampling_beta_alpha=1.5,
        time_sampling_beta_beta=1.0,
        time_sampling_scale=0.999,
        time_sampling_offset=0.001,
        min_period=4e-3,
        max_period=4.0,
        **kwargs,
    ):
        if isinstance(vlm_config, dict):
            vlm_model_type = vlm_config.get("model_type", "paligemma")
            vlm_config = CONFIG_MAPPING[vlm_model_type](**vlm_config)
        elif vlm_config is None:
            vlm_config = CONFIG_MAPPING["paligemma"](
                text_config={
                    "model_type": "gemma",
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "intermediate_size": 16384,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 1,
                    "vocab_size": 257152,
                },
                vision_config={
                    "model_type": "siglip_vision_model",
                    "intermediate_size": 4304,
                    "hidden_size": 1152,
                    "patch_size": 14,
                    "image_size": 224,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "vocab_size": 257152,
                    "vision_use_head": False,
                },
                projection_dim=2048,
            )

        if isinstance(dit_config, dict):
            dit_model_type = dit_config.get("model_type", "gemma")
            dit_config = CONFIG_MAPPING[dit_model_type](**dit_config)
        elif dit_config is None:
            dit_config = CONFIG_MAPPING["gemma"](
                hidden_size=1024,
                num_hidden_layers=18,
                intermediate_size=4096,
                num_attention_heads=8,
                num_key_value_heads=1,
                head_dim=256,
                vocab_size=vlm_config.text_config.vocab_size,
            )

        self.dit_config = dit_config
        self.vlm_config = vlm_config

        # Force bidirectional attention
        self.dit_config.is_causal = False
        self.dit_config.use_bidirectional_attention = False
        self.vlm_config.text_config.use_bidirectional_attention = True

        self.chunk_size = chunk_size
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.num_inference_steps = num_inference_steps
        self.time_sampling_beta_alpha = time_sampling_beta_alpha
        self.time_sampling_beta_beta = time_sampling_beta_beta
        self.time_sampling_scale = time_sampling_scale
        self.time_sampling_offset = time_sampling_offset
        self.min_period = min_period
        self.max_period = max_period
        super().__init__(**kwargs)


class ActionTimeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.expert_hidden_size = config.dit_config.hidden_size
        if self.expert_hidden_size % 2 != 0:
            raise ValueError(f"dimension ({self.expert_hidden_size}) must be divisible by 2")

        self.action_in_proj = nn.Linear(config.max_action_dim, self.expert_hidden_size)
        self.state_proj = nn.Linear(config.max_state_dim, self.expert_hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * self.expert_hidden_size, self.expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(self.expert_hidden_size, self.expert_hidden_size)

    def create_sinusoidal_pos_embedding(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

        min_period = self.config.min_period
        max_period = self.config.max_period
        device = time.device
        dtype = torch.float64 if device.type not in ("mps", "cpu") else torch.float32

        fraction = torch.linspace(0.0, 1.0, self.expert_hidden_size // 2, dtype=dtype, device=device)
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * time[:, None]
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

    def forward(self, state, noise, timestep):
        state_embeds = self.state_proj(state)
        action_embeds = self.action_in_proj(noise)
        time_embeds = self.create_sinusoidal_pos_embedding(timestep)
        time_embeds = time_embeds[:, None, :].expand_as(action_embeds).to(dtype=action_embeds.dtype)

        action_time_embeds = torch.cat([action_embeds, time_embeds], dim=2)
        action_time_embeds = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_embeds)))
        action_embeds_merged = torch.cat([state_embeds[:, None, :], action_time_embeds], dim=1)
        return action_embeds_merged


@auto_docstring
class PI0PreTrainedModel(PreTrainedModel):
    config: PI0Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    input_modalities = ("image", "text")


@auto_docstring
class PI0Model(PI0PreTrainedModel):
    def __init__(self, config: PI0Config):
        super().__init__(config)
        self.dit = AutoModel.from_config(config.dit_config)
        self.vlm = AutoModel.from_config(config.vlm_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vlm.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        action_embeds: torch.Tensor,  # aka `suffix_emb` (noise + state + timestep)
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,  # aka `prefix_emb` or merged image+text emb
        past_key_values: Cache | None = None,  # must-have for prefix tuning
        **kwargs,
    ) -> BaseModelOutputWithPast:
        r"""
        action_embeds (`torch.Tensor`, *optional*): args description placeholder
        pixel_attention_mask (`torch.Tensor`, *optional*): args description placeholder
        """
        if pixel_values is not None:
            # Pi0 never passes positions, so we need to infer manually
            if attention_mask is not None and position_ids is None:
                position_ids = attention_mask.cumsum(-1) - 1
            vlm_output = self.vlm(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values=None,  # `None` on purpose
                use_cache=True,
            )
            past_key_values = vlm_output.past_key_values

        # Mask for images, text and noise is bidirectional but we still need to know
        # if there are any pad tokens in the text
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("Only two-dimensional attention masks are accepted for now!")

        # Merge masks if needed, same for position ids
        # TODO: why we need `pixel_attention_mask` and can it be zero, in which cases?
        dit_position_ids = dit_attention_mask = None
        if pixel_attention_mask is not None and attention_mask is not None:
            noise_mask = torch.ones(
                action_embeds.shape[0],
                action_embeds.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            dit_attention_mask = torch.cat([attention_mask, noise_mask], dim=1)
            dit_position_ids = (torch.cumsum(dit_attention_mask, dim=1) - 1)[:, -action_embeds.shape[1]:]

        bidirectional_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=action_embeds,
            attention_mask=dit_attention_mask,
            past_key_values=past_key_values,
        )

        action_embeds = action_embeds / torch.tensor(self.config.dit_config.hidden_size**0.5, dtype=action_embeds.dtype)
        dit_output = self.dit(
            inputs_embeds=action_embeds,
            attention_mask=bidirectional_mask,
            position_ids=dit_position_ids,
            past_key_values=past_key_values,
        )
        return dit_output


class PI0ForConditionalGeneration(PI0PreTrainedModel):
    """PI0 model with action projection heads and flow matching."""

    def __init__(self, config: PI0Config):
        super().__init__(config)
        self.model = PI0Model(config)
        self.expert_hidden_size = config.dit_config.hidden_size
        self.embed_action_time = ActionTimeEmbedding(config)
        self.action_out_proj = nn.Linear(self.expert_hidden_size, config.max_action_dim)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        state: torch.FloatTensor | None = None,
        noise: torch.FloatTensor | None = None,
        timestep: torch.FloatTensor | None = None,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        actions: torch.FloatTensor = None,  # aka labels
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        actions (`torch.Tensor`, *optional*): args description placeholder
        pixel_attention_mask (`torch.Tensor`, *optional*): args description placeholder
        state  (`torch.Tensor`, *optional*): args description placeholder
        noise  (`torch.Tensor`, *optional*): args description placeholder
        timestep  (`torch.Tensor`, *optional*): args description placeholder
        """
        batch_size = state.shape[0]
        device = state.device

        # 1.Sample the timestep
        if timestep is None:
            alpha_t = torch.tensor(self.config.time_sampling_beta_alpha, dtype=torch.float32)
            beta_t = torch.tensor(self.config.time_sampling_beta_beta, dtype=torch.float32)
            dist = torch.distributions.Beta(alpha_t, beta_t)
            time_beta = dist.sample((batch_size,)).to(device)
            timestep = (time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset).float()

        # 2. Create random noise if not provided
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=device,
                dtype=pixel_values.dtype,
            )

        # 3. If training: merge noise with the ground truth actions (aka labels)
        # Target velocity is the label we want to preduct and will compute loss upon
        if actions is not None:
            time_expanded = timestep[:, None, None]
            noisy_actions = (time_expanded * noise + (1 - time_expanded) * actions).to(actions.dtype)
            target_velocity = noise - actions
        else:
            noisy_actions = noise

        # 4. Embed 'state + noise + actions' for DiT blocks
        action_time_embeds = self.embed_action_time(state, noisy_actions, timestep)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pixel_attention_mask=pixel_attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            action_embeds=action_time_embeds,
            past_key_values=past_key_values,
            **kwargs,
        )
        last_hidden_states = outputs.last_hidden_state[:, -self.config.chunk_size :]
        predicted_velocity = self.action_out_proj(last_hidden_states)

        loss = None
        if actions is not None:
            # Let the users reduce loss themselves and return fine-grained per sample loss
            loss = F.mse_loss(target_velocity, predicted_velocity, reduction="none")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=predicted_velocity,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        state: torch.FloatTensor | None = None,
        noise: torch.FloatTensor | None = None,
        num_steps: int | None = None,
    ) -> torch.FloatTensor:
        """Run flow matching inference to generate actions."""

        num_steps = num_steps or self.config.num_inference_steps
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1. Sample random noise
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=device,
                dtype=pixel_values.dtype,
            )

        # 2. Run VLM once and obtain prefix cache
        output = self.model.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values
        prefix_length = past_key_values.get_seq_length()

        # 3. Denoise `num_steps` times
        dt = -1.0 / num_steps
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(batch_size)
            output = self(
                pixel_attention_mask=pixel_attention_mask,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                state=state,
                noise=noise,
                timestep=time_tensor,
            )

            # We need to keep only the "vlm-prefix", no attention to past denoising steps!
            if past_key_values is not None:
                past_key_values.crop(prefix_length)

            noise = noise + dt * output.logits
        return noise


__all__ = [
    "PI0Config",
    "PI0PreTrainedModel",
    "PI0Model",
    "PI0ForConditionalGeneration",
    "PI0Processor",
]
