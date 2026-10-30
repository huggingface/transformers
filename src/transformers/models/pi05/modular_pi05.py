# Copyright 2026 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
"""PI0.5 model: PaliGemma + action expert conditioned on the timestep through adaptive RMS norms."""

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ..gemma.modeling_gemma import GemmaAttention, GemmaDecoderLayer, GemmaModel
from ..pi0.configuration_pi0 import PI0Config
from ..pi0.modeling_pi0 import (
    PI0ActionTimeEmbedding,
    PI0ForConditionalGeneration,
    PI0Model,
    PI0PreTrainedModel,
)
from ..pi0.processing_pi0 import PI0Processor, PI0ProcessorKwargs


logger = logging.get_logger(__name__)


class PI05ProcessorKwargs(PI0ProcessorKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "max_length": 200,
            "padding_side": "right",
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
@requires(backends=("vision", "torch"))
class PI05Processor(PI0Processor):
    valid_processor_kwargs = PI05ProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        max_state_dim: int = 32,
        chunk_size: int = 50,
        **kwargs,
    ):
        r"""
        max_state_dim (`int`, *optional*, defaults to 32):
            Size the actions are zero-padded to.
        chunk_size (`int`, *optional*, defaults to 50):
            Number of actions per chunk.
        """
        super().__init__(
            image_processor,
            tokenizer,
            chat_template=chat_template,
            max_state_dim=max_state_dim,
            chunk_size=chunk_size,
            **kwargs,
        )
        del self.state_mean
        del self.state_std
        del self.actions_mean
        del self.actions_std

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] | None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        actions: list | np.ndarray | torch.Tensor | None = None,
        state: list | np.ndarray | torch.Tensor | None = None,
        **kwargs: Unpack[PI05ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        actions (`list | np.ndarray | torch.Tensor`, *optional*):
            Actions already normalized the way the checkpoint was trained. If provided, they are padded.
        state (`list | np.ndarray | torch.Tensor`, *optional*):
            Robot state normalized to [-1, 1]. If provided, it is discretized into 256 bins in the prompt.
        """
        output_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )

        if text is None:
            logger.warning_once("You are using PI05 without a text prefix. The processor will use an empty prompt.")
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

        if state is None:
            state_prompts = [""] * len(text)
        else:
            state = torch.as_tensor(state, dtype=torch.float32).view(len(text), -1)
            discretized_state = np.digitize(state.numpy(), bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_prompts = [f", State: {' '.join(map(str, sample_state))}" for sample_state in discretized_state]
        tasks = [sample.strip().replace("_", " ").replace("\n", " ") for sample in text]
        prompt_strings = [
            f"{self.image_token * self.image_seq_length * len(image_list)}{self.tokenizer.bos_token}"
            f"Task: {task}{state_prompt};\nAction: "
            for task, state_prompt, image_list in zip(tasks, state_prompts, batched_images)
        ]
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        num_cameras = torch.tensor([len(sample_images) for sample_images in batched_images])
        pixel_attention_mask = torch.arange(num_cameras.max())[None, :] < num_cameras[:, None]
        pixel_values = self.image_processor(
            [image for sample_images in batched_images for image in sample_images],
            return_tensors="pt",
            **output_kwargs["images_kwargs"],
        )["pixel_values"]
        padded_pixel_values = pixel_values.new_zeros(*pixel_attention_mask.shape, *pixel_values.shape[1:])
        padded_pixel_values[pixel_attention_mask] = pixel_values

        model_inputs = {
            **text_inputs,
            "pixel_values": padded_pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
        }

        if actions is not None:
            actions = torch.as_tensor(actions, dtype=torch.float32)
            if actions.shape[-1] < self.max_state_dim:
                actions = F.pad(actions, (0, self.max_state_dim - actions.shape[-1]))
            model_inputs["actions"] = actions.view(-1, self.chunk_size, self.max_state_dim)

        return BatchFeature(data=model_inputs, tensor_type=return_tensors)


@auto_docstring(checkpoint="lerobot/pi05_base")
@strict
class PI05Config(PI0Config):
    r"""
    dit_config (`dict`, *optional*):
        Configuration for the action expert. Defaults to a Gemma 300M variant whose norms are conditioned on the
        timestep.
    chunk_size (`int`, *optional*, defaults to 50):
        Number of action steps to predict per chunk.
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
    >>> from transformers import PI05ForConditionalGeneration, PI05Config

    >>> config = PI05Config()
    >>> model = PI05ForConditionalGeneration(config)
    ```
    """

    model_type = "pi05"
    max_state_dim = AttributeError()


class PI05AdaptiveRMSNorm(nn.Module):
    """RMS norm whose scale, shift and residual gate are predicted from the timestep embedding."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, hidden_states: torch.Tensor, timestep_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        scale, shift, gate = self.dense(timestep_embeds)[:, None, :].chunk(3, dim=-1)
        hidden_states = hidden_states * (1.0 + scale.float()) + shift.float()
        return hidden_states.to(input_dtype), gate.to(input_dtype)


class PI05Attention(GemmaAttention):
    pass


class PI05DecoderLayer(GemmaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.input_layernorm = PI05AdaptiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PI05AdaptiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, timestep_embeds)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + gate * hidden_states

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, timestep_embeds)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate * hidden_states
        return hidden_states


class PI05ActionTimeEmbedding(PI0ActionTimeEmbedding):
    def __init__(self, config):
        super().__init__(config)
        del self.state_proj
        del self.action_time_mlp_in
        del self.action_time_mlp_out
        self.time_mlp_in = nn.Linear(config.dit_config.hidden_size, config.dit_config.hidden_size)
        self.time_mlp_out = nn.Linear(config.dit_config.hidden_size, config.dit_config.hidden_size)

    def forward(self, noise, timestep):
        action_embeds = self.action_in_proj(noise)
        timestep_embeds = self.sinusoid_embeds(timestep).to(dtype=action_embeds.dtype)
        timestep_embeds = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(timestep_embeds))))
        return action_embeds, timestep_embeds


@auto_docstring
class PI05PreTrainedModel(PI0PreTrainedModel):
    main_input_name = "input_ids"
    _no_split_modules = ["PI05DecoderLayer"]
    _can_record_outputs = {"hidden_states": PI05DecoderLayer, "attentions": PI05Attention}
    _keys_to_ignore_on_load_unexpected = [r"model\.dit\.embed_tokens"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, PI05AdaptiveRMSNorm):
            init.zeros_(module.dense.weight)
            init.zeros_(module.dense.bias)


@auto_docstring
class PI05ActionExpertModel(GemmaModel):
    def __init__(self, config):
        super().__init__(config)
        del self.embed_tokens
        self.norm = PI05AdaptiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        timestep_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        timestep_embeds (`torch.Tensor` of shape `(batch_size, hidden_size)`):
            Embedding of the denoising timestep, used to modulate every norm of the action expert.
        """
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                timestep_embeds=timestep_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states, _ = self.norm(hidden_states, timestep_embeds)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring
class PI05Model(PI0Model):
    def __init__(self, config: PI05Config):
        super().__init__(config)
        self.dit = PI05ActionExpertModel(config.dit_config)


@auto_docstring
class PI05ForConditionalGeneration(PI0ForConditionalGeneration):
    """PI0.5 model with action projection heads and flow matching."""

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        noise: torch.FloatTensor | None = None,
        timestep: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        actions: torch.FloatTensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        noise (`torch.Tensor`, *optional*):
            Noisy actions at the current timestep. Sampled from a normal distribution when `actions` are given.
        timestep (`torch.Tensor`, *optional*):
            Current denoising timestep.
        pixel_attention_mask (`torch.Tensor`, *optional*):
            The mask indicating padded positions in the input image.
        actions (`torch.Tensor`, *optional*):
            Input actions that need to be predicted. Used only when training to compute loss.
        """
        if noise is None:
            noise = torch.randn(
                input_ids.shape[0],
                self.config.chunk_size,
                self.config.max_action_dim,
                device=input_ids.device,
                dtype=self.dtype,
            )
        batch_size = noise.shape[0]

        if timestep is None:
            alpha_t = torch.tensor(self.config.time_sampling_beta_alpha, dtype=torch.float32)
            beta_t = torch.tensor(self.config.time_sampling_beta_beta, dtype=torch.float32)
            dist = torch.distributions.Beta(alpha_t, beta_t)
            time_beta = dist.sample((batch_size,)).to(noise.device)
            timestep = (time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset).float()

        if actions is not None:
            time_expanded = timestep[:, None, None]
            noisy_actions = (time_expanded * noise + (1 - time_expanded) * actions).to(actions.dtype)
            target_velocity = noise - actions
        else:
            noisy_actions = noise

        action_embeds, timestep_embeds = self.embed_action_time(noisy_actions, timestep)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pixel_attention_mask=pixel_attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            action_embeds=action_embeds,
            timestep_embeds=timestep_embeds,
            past_key_values=past_key_values,
            **kwargs,
        )
        last_hidden_states = outputs.last_hidden_state[:, -self.config.chunk_size :]
        predicted_velocity = self.action_out_proj(last_hidden_states)

        loss = None
        if actions is not None:
            loss = F.mse_loss(target_velocity, predicted_velocity, reduction=self.config.loss_reduction)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=predicted_velocity,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "PI05Config",
    "PI05PreTrainedModel",
    "PI05ActionExpertModel",
    "PI05Model",
    "PI05ForConditionalGeneration",
    "PI05Processor",
]
