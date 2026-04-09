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
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessingKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, can_return_tuple, logging
from ...utils.generic import maybe_autocast
from ...utils.import_utils import requires
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..paligemma.processing_paligemma import PaligemmaProcessor
from ..siglip.image_processing_siglip import SiglipImageProcessor


logger = logging.get_logger(__name__)


@auto_docstring
class PI0ImageProcessor(SiglipImageProcessor):
    size = {"max_height": 224, "max_width": 224}
    pad_size = {"height": 224, "width": 224}
    do_pad = True


class PI0ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "max_length": 48,
            "padding_side": "right",
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
@requires(backends=("vision", "torch"))
class PI0Processor(PaligemmaProcessor):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.height, self.width = image_processor.size["height"], image_processor.size["width"]
        state_mean = kwargs.get("state_mean", [-0.0419, 0.0354, 0.8257, 2.9083, -0.5562, -0.1665, 0.0283, -0.0286])
        state_std = kwargs.get("state_std", [0.1074, 0.1442, 0.2572, 0.3441, 1.2344, 0.3580, 0.0133, 0.0132])
        actions_mean = kwargs.get("actions_mean", [0.0182, 0.0586, -0.0559, 0.0046, 0.0029, -0.0077, -0.0916])
        actions_std = kwargs.get("actions_std", [0.2825, 0.3590, 0.3674, 0.0377, 0.0543, 0.0872, 0.9958])

        self.state_mean = torch.tensor(state_mean)
        self.state_std = torch.tensor(state_std)
        self.actions_mean = torch.tensor(actions_mean)
        self.actions_std = torch.tensor(actions_std)
        self.max_state_dim = kwargs.get("max_state_dim", 32)
        self.chunk_size = kwargs.get("chunk_size", 50)
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] | None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        actions: list | np.ndarray | torch.Tensor | None = None,
        state: list | np.ndarray | torch.Tensor | None = None,
        **kwargs: Unpack[PI0ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        actions (`list | np.ndarray | torch.Tensor`, *optional*):
            Actions to be predicted by the model. If provided, padding, mean and std normalization will be applied.
        state (`list | np.ndarray | torch.Tensor`, *optional*):
            Robotic states to be predicted by the model. If provided, padding, mean and std normalization will be applied.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`. If `suffix`
              is provided, the `input_ids` will also contain the suffix input ids.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_attention_mask** -- Pixel values padding mask to be fed to a model. Returned when `images` is not `None`.
            - **state** -- Robot state compatible with model if `state` is not None
            - **actions** -- Label-actions compatible with training if `actions` is not None
        """
        output_kwargs = self._merge_kwargs(
            PI0ProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )

        if text is None:
            logger.warning_once("You are using PI0 without a text prefix. The processor will use an empty prompt.")
            text = ""

        if isinstance(text, str):
            text = [text]

        batched_images = make_nested_list_of_images(images)
        if len(batched_images) != len(text):
            raise ValueError(
                f"Received {len(batched_images)} image samples for {len(text)} prompts. "
                "Each prompt should be associated with one sample (with one or more camera images)."
            )

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        output_kwargs["images_kwargs"].pop("return_tensors", None)

        prompt_strings = []
        for sample, image_list in zip(text, batched_images):
            sample = (
                f"{self.image_token * self.image_seq_length * len(image_list)}{self.tokenizer.bos_token}{sample}\n"
            )
            prompt_strings.append(sample)

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        # Here is the diff from PaliGemma. Ideally we'd create a new ImageProcessor if it were a VLM
        max_num_cameras = max(len(sample_images) for sample_images in batched_images)
        pixel_attention_mask = torch.zeros((len(batched_images), max_num_cameras), dtype=torch.bool)
        padded_pixel_values = torch.zeros(len(batched_images), max_num_cameras, 3, self.height, self.width)

        for batch, sample_images in enumerate(batched_images):
            processed = self.image_processor(sample_images, return_tensors="pt", **output_kwargs["images_kwargs"])

            num_cameras = len(sample_images)
            pixel_attention_mask[batch, :num_cameras] = True
            padded_pixel_values[batch, :num_cameras] = processed["pixel_values"]

        return_data = {
            **text_inputs,
            "pixel_values": padded_pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
        }

        if actions is not None:
            actions = (torch.tensor(actions) - self.actions_mean) / (self.actions_std + 1e-08)
            if actions.shape[-1] < self.max_state_dim:
                actions = F.pad(actions, (0, self.max_state_dim - actions.shape[-1]))
            return_data["actions"] = actions.view(-1, self.chunk_size, self.max_state_dim)

        if state is not None:
            state = (torch.tensor(state) - self.state_mean) / (self.state_std + 1e-08)
            if state.shape[-1] < self.max_state_dim:
                state = F.pad(state, (0, self.max_state_dim - state.shape[-1]))
            return_data["state"] = state.view(-1, self.max_state_dim)

        return BatchFeature(data=return_data, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        return super().model_input_names + ["pixel_attention_mask"]


@auto_docstring(checkpoint="lerobot/pi0_base")
@strict
class PI0Config(PreTrainedConfig):
    r"""
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
    loss_reduction (`str`, *optional*, defaults to `"mean"`):
        The reduction to use on MSE loss.

    Example:
    ```python
    >>> from transformers import PI0ForConditionalGeneration, PI0Config

    >>> config = PI0Config()
    >>> model = PI0ForConditionalGeneration(config)
    ```
    """

    model_type = "pi0"
    sub_configs = {"vlm_config": AutoConfig, "dit_config": AutoConfig}

    vlm_config: dict | PreTrainedConfig | None = None
    dit_config: dict | PreTrainedConfig | None = None
    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    loss_reduction: str = "mean"

    def __post_init__(self, **kwargs):
        if isinstance(self.vlm_config, dict):
            vlm_model_type = self.vlm_config.get("model_type", "paligemma")
            self.vlm_config = CONFIG_MAPPING[vlm_model_type](**self.vlm_config)
        elif self.vlm_config is None:
            self.vlm_config = CONFIG_MAPPING["paligemma"](
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
                image_token_id=257152,
            )

        if isinstance(self.dit_config, dict):
            dit_model_type = self.dit_config.get("model_type", "gemma")
            self.dit_config = CONFIG_MAPPING[dit_model_type](**self.dit_config)
        elif self.dit_config is None:
            self.dit_config = CONFIG_MAPPING["gemma"](
                hidden_size=1024,
                num_hidden_layers=18,
                intermediate_size=4096,
                num_attention_heads=8,
                num_key_value_heads=1,
                head_dim=256,
                vocab_size=self.vlm_config.text_config.vocab_size,
            )

        # Force bidirectional attention
        self.dit_config.is_causal = False
        self.dit_config.use_bidirectional_attention = True
        self.vlm_config.text_config.use_bidirectional_attention = True
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.dit_config.hidden_size % 2 != 0:
            raise ValueError(f"DiT hidden dim=({self.config.dit_config.hidden_size}) must be divisible by 2")


def blockwise_bidirectional_mask(block_boundaries: torch.Tensor) -> Callable:
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        q_block = torch.bucketize(q_idx, block_boundaries)
        kv_block = torch.bucketize(kv_idx, block_boundaries)
        return kv_block <= q_block

    return inner_mask


class PI0TimestepEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        sinusoid_freq = self.compute_freqs(config)
        self.register_buffer("sinusoid_freq", sinusoid_freq, persistent=False)

    @staticmethod
    def compute_freqs(config):
        fraction = torch.linspace(0.0, 1.0, config.dit_config.hidden_size // 2, dtype=torch.float32)
        period = config.min_period * (config.max_period / config.min_period) ** fraction
        sinusoid_freq = 1.0 / period * 2 * math.pi
        return sinusoid_freq

    def forward(self, time):
        device_type = time.device.type if isinstance(time.device.type, str) and time.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            sinusoid_freq = self.sinusoid_freq[None, :]
            emb = sinusoid_freq * time[:, None]
            time_embeds = torch.cat([emb.sin(), emb.cos()], dim=1)
        return time_embeds


class PI0ActionTimeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sinusoid_embeds = PI0TimestepEmbeddings(config)
        self.action_in_proj = nn.Linear(config.max_action_dim, config.dit_config.hidden_size)
        self.state_proj = nn.Linear(config.max_state_dim, config.dit_config.hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * config.dit_config.hidden_size, config.dit_config.hidden_size)
        self.action_time_mlp_out = nn.Linear(config.dit_config.hidden_size, config.dit_config.hidden_size)

    def forward(self, state, noise, timestep):
        state_embeds = self.state_proj(state)
        action_embeds = self.action_in_proj(noise)

        time_embeds = self.sinusoid_embeds(timestep)
        time_embeds = time_embeds[:, None, :].expand_as(action_embeds).to(dtype=action_embeds.dtype)

        action_time_embeds = torch.cat([action_embeds, time_embeds], dim=2)
        action_time_embeds = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_embeds)))
        action_embeds_merged = torch.cat([state_embeds[:, None, :], action_time_embeds], dim=1)
        return action_embeds_merged


@auto_docstring
class PI0PreTrainedModel(PreTrainedModel):
    config: PI0Config
    base_model_prefix = "model"
    main_input_name = "state"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    input_modalities = ("image", "text")

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, PI0TimestepEmbeddings):
            init.copy_(module.sinusoid_freq, module.compute_freqs(module.config))


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

    def embed_prefix(self, input_ids, pixel_values, pixel_attention_mask, attention_mask=None):
        max_num_cameras = pixel_attention_mask.shape[1]
        pixel_values = pixel_values.flatten(0, 1)
        image_features = self.vlm.get_image_features(pixel_values).pooler_output
        image_features = image_features.reshape(-1, max_num_cameras, image_features.shape[1], image_features.shape[2])

        total_image_features = []
        for batch_idx, mask in enumerate(pixel_attention_mask):
            unpadded_image_features = image_features[batch_idx][mask]
            total_image_features.append(unpadded_image_features)
        total_image_features = torch.cat(total_image_features, dim=0)

        llm_input_ids = input_ids.clone()
        llm_input_ids[input_ids == self.config.vlm_config.image_token_id] = 0
        inputs_embeds = self.vlm.get_input_embeddings()(llm_input_ids)
        special_image_mask = (
            (input_ids == self.config.vlm_config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, total_image_features)

        return inputs_embeds

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
        action_embeds (`torch.Tensor`, *optional*):
            The embeddings of input actions and robot states.
        pixel_attention_mask (`torch.Tensor`, *optional*):
            The mask indicating padded positions in the input image.
        """
        if pixel_values is not None and past_key_values is None:
            if attention_mask is not None and position_ids is None:
                position_ids = attention_mask.cumsum(-1) - 1

            if inputs_embeds is None:
                inputs_embeds = self.embed_prefix(input_ids, pixel_values, pixel_attention_mask)

            token_type_ids = torch.zeros_like(inputs_embeds)[:, :, 0]
            past_key_values = self.vlm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                use_cache=True,
            ).past_key_values

        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("Only two-dimensional attention masks are accepted for now!")

        # Merge masks if needed, same for position ids
        dit_position_ids = dit_attention_mask = None
        if attention_mask is not None:
            noise_mask = torch.ones(
                action_embeds.shape[0],
                action_embeds.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            dit_attention_mask = torch.cat([attention_mask, noise_mask], dim=1)
            dit_position_ids = (torch.cumsum(dit_attention_mask, dim=1) - 1)[:, -action_embeds.shape[1] :]

        # We have three blocks: vlm-inputss, state and actions from which only 1 token is `state`
        # The mask should be bidirectional within each block and to prev blocks, but not to next blocks
        vlm_input_length = past_key_values.get_seq_length()
        block_sizes = torch.tensor([vlm_input_length + 1, action_embeds.shape[1] - 1], device=action_embeds.device)
        block_boundaries = torch.cumsum(block_sizes, dim=0) - 1
        bidirectional_mask = create_bidirectional_mask(
            config=self.config.dit_config,
            inputs_embeds=action_embeds,
            attention_mask=dit_attention_mask,
            past_key_values=past_key_values,
            and_mask_function=blockwise_bidirectional_mask(block_boundaries),
        )

        dit_output = self.dit(
            inputs_embeds=action_embeds,
            attention_mask=bidirectional_mask,
            position_ids=dit_position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        return dit_output


class PI0ForConditionalGeneration(PI0PreTrainedModel):
    """PI0 model with action projection heads and flow matching."""

    _tp_plan = {"action_out_proj": "colwise_gather_output"}

    def __init__(self, config: PI0Config):
        super().__init__(config)
        self.model = PI0Model(config)
        self.expert_hidden_size = config.dit_config.hidden_size
        self.embed_action_time = PI0ActionTimeEmbedding(config)
        self.action_out_proj = nn.Linear(self.expert_hidden_size, config.max_action_dim)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        state: torch.FloatTensor,
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
        state (`torch.Tensor`, *optional*):
            Current robot state.
        noise (`torch.Tensor`, *optional*):
            Random noise at current timestep that needs to be denoised
        timestep (`torch.Tensor`, *optional*):
            Current denoising timestep.
        pixel_attention_mask (`torch.Tensor`, *optional*):
            The mask indicating padded positions in the input image.
        actions (`torch.Tensor`, *optional*):
            Input actions that need to be predicted. Used only when training to compiute loss.
        """
        batch_size = state.shape[0]

        # 1.Sample the timestep
        if timestep is None:
            alpha_t = torch.tensor(self.config.time_sampling_beta_alpha, dtype=torch.float32)
            beta_t = torch.tensor(self.config.time_sampling_beta_beta, dtype=torch.float32)
            dist = torch.distributions.Beta(alpha_t, beta_t)
            time_beta = dist.sample((batch_size,)).to(state.device)
            timestep = (time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset).float()

        # 2. Create random noise if not provided
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=state.device,
                dtype=state.dtype,
            )

        # 3. If training: merge noise with the ground truth actions (aka labels)
        # Target velocity is the label we want to predict and will compute loss upon
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
            loss = F.mse_loss(target_velocity, predicted_velocity, reduction=self.config.loss_reduction)

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
        state: torch.FloatTensor,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        noise: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        num_steps: int | None = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """Run flow matching inference to generate actions."""

        num_steps = num_steps or self.config.num_inference_steps
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1. Sample random noise
        if noise is None:
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=(
                    batch_size,
                    self.config.chunk_size,
                    self.config.max_action_dim,
                ),
                dtype=pixel_values.dtype,
                device=device,
            )

        # 2. Run VLM once and obtain prefix cache. Must infer positions here!
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(-1) - 1
        inputs_embeds = self.model.embed_prefix(input_ids, pixel_values, pixel_attention_mask)
        past_key_values = self.model.vlm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        ).past_key_values
        prefix_length = past_key_values.get_seq_length()

        # 3. Denoise `num_steps` times
        dt = -1.0 / num_steps
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(batch_size)
            output = self(
                state=state,
                noise=noise,
                timestep=time_tensor,
                pixel_attention_mask=pixel_attention_mask,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

            # We need to keep only the "vlm-prefix", no attention to past denoising steps!
            past_key_values.crop(prefix_length)
            noise = noise + dt * output.logits
        return noise


__all__ = [
    "PI0Config",
    "PI0PreTrainedModel",
    "PI0Model",
    "PI0ForConditionalGeneration",
    "PI0Processor",
    "PI0ImageProcessor",
]
