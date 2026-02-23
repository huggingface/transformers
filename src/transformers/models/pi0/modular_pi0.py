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
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...masking_utils import create_bidirectional_mask, create_causal_mask, or_masks
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import ModelOutput, auto_docstring, logging
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..auto import AutoConfig, AutoModel
from ..gemma.modeling_gemma import GemmaAttention, GemmaDecoderLayer, apply_rotary_pos_emb, eager_attention_forward
from ..paligemma.configuration_paligemma import PaliGemmaConfig
from ..paligemma.modeling_paligemma import PaliGemmaMultiModalProjector
from ..siglip import SiglipVisionConfig


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


class PI0Config(PaliGemmaConfig):
    r"""
    Configuration class for PI0.

    PI0 is a robot action prediction model that combines a PaliGemma VLM backbone
    with an action expert Gemma model. It uses flow matching for continuous action generation.

    This model inherits from [`PaliGemmaConfig`]. See the superclass documentation for more details.
    Example checkpoint: [lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base).

    Args:
        vision_config (`dict`, *optional*):
            Configuration for the vision encoder (SiglipVisionModel).
        text_config (`dict`, *optional*):
            Configuration for the language model (GemmaModel).
        expert_config (`dict`, *optional*):
            Configuration for the action expert (GemmaModel). Defaults to a Gemma 300M variant.
        image_token_index (`int`, *optional*, defaults to 256000):
            The image token index to encode the image prompt.
        vocab_size (`int`, *optional*, defaults to 257152):
            Vocabulary size of the PI0 language backbone.
        projection_dim (`int`, *optional*, defaults to 2048):
            Dimension of the multimodal projection space.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden layer of the language model.
        tie_word_embeddings (`bool | None`, *optional*, defaults to `True`):
            Whether to tie word embeddings.
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
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": SiglipVisionConfig,
        "expert_config": AutoConfig,
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        expert_config=None,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        tie_word_embeddings: bool | None = True,
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
        if vision_config is None:
            vision_config = {
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "patch_size": 14,
                "image_size": 224,
                "vision_use_head": False,
            }
        if isinstance(text_config, dict):
            text_vocab_size = text_config.get("vocab_size", 257152)
        elif text_config is not None:
            text_vocab_size = text_config.vocab_size
        else:
            text_vocab_size = 257152

        if isinstance(expert_config, dict):
            expert_config["model_type"] = expert_config.get("model_type", "gemma")
            self.expert_config = AutoConfig.for_model(**expert_config)
        elif expert_config is None:
            self.expert_config = AutoConfig.for_model(
                "gemma",
                hidden_size=1024,
                num_hidden_layers=18,
                intermediate_size=4096,
                num_attention_heads=8,
                num_key_value_heads=1,
                head_dim=256,
                vocab_size=text_vocab_size,
            )
        else:
            self.expert_config = expert_config

        self.expert_config.is_causal = False
        self.expert_config.use_bidirectional_attention = True

        super().__init__(
            vision_config=vision_config,
            text_config=text_config,
            image_token_index=image_token_index,
            vocab_size=vocab_size,
            projection_dim=projection_dim,
            hidden_size=hidden_size,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = self.text_config.vocab_size

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


@dataclass
class PI0Output(ModelOutput):
    """Output type for PI0ForConditionalGeneration.

    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Flow matching MSE loss, returned when `actions` is provided.
    loss_per_sample (`torch.FloatTensor` of shape `(batch_size, chunk_size, max_action_dim)`, *optional*):
        Per-element MSE loss before reduction, returned when `actions` is provided.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Hidden states of the action expert suffix sequence when `output_hidden_states=True`.
    attentions (`tuple(torch.FloatTensor)`, *optional*):
        Attention maps of the action expert suffix sequence when `output_attentions=True`.
    """

    loss: torch.FloatTensor | None = None
    loss_per_sample: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float
) -> torch.Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    device = time.device
    dtype = torch.float64 if device.type not in ("mps", "cpu") else torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, batch_size: int, device: torch.device) -> torch.Tensor:
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((batch_size,)).to(device)


class PI0MultiModalProjector(PaliGemmaMultiModalProjector):
    pass


@auto_docstring
class PI0PreTrainedModel(PreTrainedModel):
    config_class = PI0Config
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["PI0MultiModalProjector", "GemmaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True


def compute_layer_complete(
    layer_idx: int,
    inputs_embeds: list[torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    language_model: nn.Module,
    expert_model: nn.Module,
) -> list[torch.Tensor]:
    """Compute one merged attention layer across VLM language model and action expert."""
    models = [language_model, expert_model]
    query_states = []
    key_states = []
    value_states = []

    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states = layer.input_layernorm(hidden_states)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    cos, sin = language_model.rotary_emb(
        torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[-1],
            device=query_states.device,
            dtype=query_states.dtype,
        ),
        position_ids,
    )
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

    vlm_layer = language_model.layers[layer_idx]
    attn_implementation = vlm_layer.self_attn.config._attn_implementation
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(attn_implementation, eager_attention_forward)
    att_output, attn_weights = attention_interface(
        vlm_layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=vlm_layer.self_attn.scaling,
    )

    batch_size = query_states.shape[0]
    head_dim = vlm_layer.self_attn.head_dim
    num_heads = vlm_layer.self_attn.config.num_attention_heads
    att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        current_att = att_output[:, start_pos:end_pos]
        if current_att.dtype != layer.self_attn.o_proj.weight.dtype:
            current_att = current_att.to(layer.self_attn.o_proj.weight.dtype)
        precomputed_attn_output = layer.self_attn.o_proj(current_att)

        if i > 0:
            # Expert: call GemmaDecoderLayer.forward() so _can_record_outputs hooks fire naturally.
            # Inject the joint-attention result by hooking self_attn before the layer call.
            # super hacky, Arthur wdyt?
            def make_attn_hook(attn_out, weights):
                def hook(module, args, output):
                    return (attn_out, weights)

                return hook

            handle = layer.self_attn.register_forward_hook(
                make_attn_hook(precomputed_attn_output, attn_weights), prepend=True
            )
            outputs_embeds.append(layer(hidden_states))
            handle.remove()
        else:
            # Language model: manual residual + MLP (no recording needed).
            hidden_states = hidden_states + precomputed_attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            outputs_embeds.append(residual + hidden_states)

        start_pos = end_pos

    return outputs_embeds


class PI0Model(PI0PreTrainedModel):
    """PI0 backbone: vision tower + language model + action expert with merged attention."""

    def __init__(self, config: PI0Config):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = PI0MultiModalProjector(config)
        self.language_model = AutoModel.from_config(config=config.text_config)
        self.expert = AutoModel.from_config(config=config.expert_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, num_cameras = pixel_values.shape[:2]
        pixel_values = pixel_values.reshape(batch_size * num_cameras, *pixel_values.shape[2:])
        image_outputs = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(image_outputs.last_hidden_state)
        image_features = image_features.reshape(
            batch_size, num_cameras, image_features.shape[1], image_features.shape[2]
        )
        return image_features.flatten(1, 2)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        lang_emb = self.language_model.embed_tokens(tokens)
        return lang_emb * math.sqrt(lang_emb.shape[-1])

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Cache | None,
        inputs_embeds: list[torch.Tensor],
        use_cache: bool = False,
    ) -> tuple[list[torch.Tensor | None], Cache | None]:
        # Prefix-only path (cache the VLM prefix for inference)
        if inputs_embeds[1] is None:
            bidirectional_mask = create_bidirectional_mask(self.config.text_config, inputs_embeds[0], attention_mask)
            prefix_output = self.language_model(
                inputs_embeds=inputs_embeds[0],
                attention_mask=bidirectional_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            return [prefix_output.last_hidden_state, None], prefix_output.past_key_values

        # Suffix-only path (use cached prefix KV)
        if inputs_embeds[0] is None:
            suffix_output = self.expert(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            return [None, suffix_output.last_hidden_state], None

        # Full merged attention path (training)
        num_layers = self.config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            inputs_embeds = compute_layer_complete(
                layer_idx,
                inputs_embeds,
                attention_mask,
                position_ids,
                language_model=self.language_model,
                expert_model=self.expert,
            )

        # Final norms
        prefix_output = self.language_model.norm(inputs_embeds[0])
        suffix_output = self.expert.norm(inputs_embeds[1])

        return [prefix_output, suffix_output], None


class PI0ForConditionalGeneration(PI0PreTrainedModel):
    """PI0 model with action projection heads and flow matching."""

    _can_record_outputs = {
        "hidden_states": GemmaDecoderLayer,
        "attentions": GemmaAttention,
    }
    main_input_name = "pixel_values"

    def __init__(self, config: PI0Config):
        super().__init__(config)
        self.model = PI0Model(config)
        self.vocab_size = config.vocab_size

        expert_hidden_size = config.expert_config.hidden_size

        self.action_in_proj = nn.Linear(config.max_action_dim, expert_hidden_size)
        self.action_out_proj = nn.Linear(expert_hidden_size, config.max_action_dim)
        self.state_proj = nn.Linear(config.max_state_dim, expert_hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * expert_hidden_size, expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(expert_hidden_size, expert_hidden_size)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def embed_prefix(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_masks: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed images and language tokens into a prefix sequence."""
        batch_size = input_ids.shape[0]
        embs = []
        pad_masks = []

        image_features = self.model.get_image_features(pixel_values)

        if image_masks is not None:
            num_cameras = image_masks.shape[1]
            image_seq_len = image_features.shape[1] // num_cameras
            for cam_idx in range(num_cameras):
                cam_features = image_features[:, cam_idx * image_seq_len : (cam_idx + 1) * image_seq_len]
                embs.append(cam_features)
                cam_mask = image_masks[:, cam_idx]
                pad_masks.append(cam_mask[:, None].expand(batch_size, image_seq_len))
        else:
            num_image_tokens = image_features.shape[1]
            embs.append(image_features)
            pad_masks.append(torch.ones(batch_size, num_image_tokens, dtype=torch.bool, device=image_features.device))

        lang_emb = self.model.embed_language_tokens(input_ids)
        embs.append(lang_emb)
        if attention_mask is None:
            pad_masks.append(torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device))
        else:
            pad_masks.append(attention_mask.bool())

        return torch.cat(embs, dim=1), torch.cat(pad_masks, dim=1)

    def embed_suffix(
        self,
        state: torch.FloatTensor,
        noisy_actions: torch.FloatTensor,
        timestep: torch.FloatTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed suffix: state + action-time fusion."""
        expert_hidden_size = self.action_in_proj.out_features

        state_emb = self.state_proj(state)
        batch_size = state_emb.shape[0]
        device = state_emb.device

        time_emb = create_sinusoidal_pos_embedding(
            timestep, expert_hidden_size, min_period=self.config.min_period, max_period=self.config.max_period
        )

        action_emb = self.action_in_proj(noisy_actions)
        time_emb_expanded = time_emb[:, None, :].expand_as(action_emb).to(dtype=action_emb.dtype)
        action_time_emb = torch.cat([action_emb, time_emb_expanded], dim=2)
        action_time_emb = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_emb)))

        embs = torch.cat([state_emb[:, None, :], action_time_emb], dim=1)
        pad_masks = torch.ones(batch_size, embs.shape[1], dtype=torch.bool, device=device)

        return embs, pad_masks

    @capture_outputs
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        state: torch.FloatTensor | None = None,
        actions: torch.FloatTensor | None = None,
        image_masks: torch.BoolTensor | None = None,
        noise: torch.FloatTensor | None = None,
        timestep: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> PI0Output:
        """Training forward pass with flow matching loss."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if actions is None:
            raise ValueError("actions must be provided for training. Use sample_actions for inference.")
        del labels  # Unused; accepted for compatibility with generic testing utilities.
        del return_dict  # Handled by `capture_outputs`.

        if noise is None:
            noise = torch.randn_like(actions)
        elif noise.shape != actions.shape:
            noise = torch.zeros_like(actions)
        if timestep is None:
            time_beta = sample_beta(
                self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, batch_size, device
            )
            timestep = (time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset).float()

        time_expanded = timestep[:, None, None]
        noisy_actions = (time_expanded * noise + (1 - time_expanded) * actions).to(actions.dtype)
        target_velocity = noise - actions

        prefix_embs, prefix_pad_masks = self.embed_prefix(pixel_values, input_ids, attention_mask, image_masks)
        suffix_embs, suffix_pad_masks = self.embed_suffix(state, noisy_actions, timestep)

        if prefix_embs.dtype != suffix_embs.dtype:
            suffix_embs = suffix_embs.to(dtype=prefix_embs.dtype)

        combined_pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        prefix_length = prefix_embs.shape[1]
        total_length = prefix_length + suffix_embs.shape[1]

        position_ids = torch.cumsum(combined_pad_masks, dim=1) - 1
        cache_position = torch.arange(total_length, device=device)

        def prefix_bidirectional(batch_idx, head_idx, q_idx, kv_idx):
            return (q_idx < prefix_length) & (kv_idx < prefix_length)

        def suffix_bidirectional(batch_idx, head_idx, q_idx, kv_idx):
            return (q_idx >= prefix_length) & (kv_idx >= prefix_length)

        attention_mask_4d = create_causal_mask(
            config=self.config.text_config,
            inputs_embeds=prefix_embs,
            attention_mask=combined_pad_masks,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
            or_mask_function=or_masks(prefix_bidirectional, suffix_bidirectional),
        )

        inputs_embeds = [prefix_embs, suffix_embs]
        num_layers = self.config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            inputs_embeds = compute_layer_complete(
                layer_idx,
                inputs_embeds,
                attention_mask_4d,
                position_ids,
                language_model=self.model.language_model,
                expert_model=self.model.expert,
            )

        suffix_out = self.model.expert.norm(inputs_embeds[1])

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        predicted_velocity = self.action_out_proj(suffix_out)

        loss_per_sample = F.mse_loss(target_velocity, predicted_velocity, reduction="none")
        loss = loss_per_sample.mean()

        return PI0Output(loss=loss, loss_per_sample=loss_per_sample)

    @torch.no_grad()
    def sample_actions(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        state: torch.FloatTensor | None = None,
        image_masks: torch.BoolTensor | None = None,
        noise: torch.FloatTensor | None = None,
        num_steps: int | None = None,
    ) -> torch.FloatTensor:
        """Run flow matching inference to generate actions."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        batch_size = input_ids.shape[0]
        device = input_ids.device

        if noise is None:
            noise = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=device,
                dtype=pixel_values.dtype,
            )

        prefix_embs, prefix_pad_masks = self.embed_prefix(pixel_values, input_ids, attention_mask, image_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.model(
            attention_mask=prefix_pad_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        prefix_length = prefix_embs.shape[1]

        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(batch_size)

            suffix_embs, suffix_pad_masks = self.embed_suffix(state, x_t, time_tensor)

            full_pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            (_, suffix_out), _ = self.model(
                attention_mask=full_pad_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
            )
            if past_key_values is not None:
                past_key_values.crop(prefix_length)

            suffix_out = suffix_out[:, -self.config.chunk_size :]
            v_t = self.action_out_proj(suffix_out)

            x_t = x_t + dt * v_t

        return x_t


__all__ = [
    "PI0Config",
    "PI0PreTrainedModel",
    "PI0Model",
    "PI0ForConditionalGeneration",
    "PI0Processor",
]
