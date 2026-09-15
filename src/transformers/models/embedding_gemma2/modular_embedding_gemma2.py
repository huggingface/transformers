# Copyright 2026 the HuggingFace Team. All rights reserved.
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
from collections import UserDict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from ... import initialization as init
from ...audio_utils import AudioInput
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...image_utils import ImageInput, make_nested_list_of_images
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring, is_vision_available, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...video_processing_utils import BaseVideoProcessor, VideoMetadata
from ...video_utils import VideoInput, make_batched_videos
from ..auto.modeling_auto import AutoModel
from ..gemma4 import Gemma4AudioConfig, Gemma4VisionConfig
from ..gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig
from ..gemma4.modeling_gemma4 import (
    Gemma4Model,
    Gemma4MultimodalEmbedder,
    Gemma4PreTrainedModel,
    Gemma4RMSNorm,
    Gemma4TextDecoderLayer,
    Gemma4TextExperts,
    Gemma4TextModel,
    Gemma4TextModelOutputWithPast,
    Gemma4TextRotaryEmbedding,
    Gemma4TextRouter,
    Gemma4TextScaledWordEmbedding,
)
from ..gemma4.processing_gemma4 import Gemma4Processor, Gemma4ProcessorKwargs
from ..gemma4.video_processing_gemma4 import Gemma4VideoProcessor, Gemma4VideoProcessorKwargs


if is_vision_available():
    from ..gemma4.image_processing_gemma4 import (
        Gemma4ImageProcessorKwargs,
        # Only the inlined Gemma 4 processor code uses this, so it looks unused here. Dropping it
        # makes the converter emit an import from `image_processing_embedding_gemma2`, a module
        # this model deliberately never generates because it reuses the Gemma 4 image processor.
        # trf-ignore: TRF039
        get_aspect_ratio_preserving_size,  # noqa: F401
    )


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/embeddinggemma-2")
class EmbeddingGemma2TextConfig(Gemma4TextConfig):
    r"""
    embedding_dim (`int`, *optional*, defaults to 768):
        Dimensionality of the pooled sentence embedding produced by `embedding_projection`.
    hidden_size_per_layer_input (`int`, *optional*, defaults to 512):
        Dimensionality of the per-layer (PLE) residual signal. EmbeddingGemma 2 uses
        *projection-only* PLE: the signal is derived from `inputs_embeds` alone, with no
        auxiliary token lookup table, hence `vocab_size_per_layer_input` does not exist.
    use_bidirectional_attention (`str`, *optional*, defaults to `"all"`):
        EmbeddingGemma 2 attends bidirectionally over the full sequence, so the stack behaves
        as an encoder despite reusing the Gemma 4 decoder-layer implementation.
    attention_k_eq_v (`bool`, defaults to `False`):
        Whether keys and values share the same projection weights. When `True`, the key
        projection output is reused as the value projection. Unused by the released
        EmbeddingGemma 2 checkpoints.
    num_kv_shared_layers (`int`, defaults to 0):
        Number of consecutive decoder layers that share the same key-value projections.
        A value of 0 means no sharing (each layer has independent KV projections). Unused by
        the released EmbeddingGemma 2 checkpoints.
    enable_moe_block (`bool`, defaults to `False`):
        Whether to enable Mixture-of-Experts (MoE) blocks in the decoder layers. When
        `True`, eligible layers will use a sparse MoE feed-forward network. Unused by the
        released EmbeddingGemma 2 checkpoints.
    use_double_wide_mlp (`bool`, defaults to `False`):
        Whether to use a double-width MLP with fused gate and up projections. Unused by the
        released EmbeddingGemma 2 checkpoints.
    top_k_experts (`int`, *optional*):
        Number of experts activated per token in MoE layers. Only used when
        `enable_moe_block=True`.
    moe_intermediate_size (`int`, *optional*):
        Intermediate (hidden) size of each expert's feed-forward network in MoE layers.
        Only used when `enable_moe_block=True`.
    """

    model_type = "embedding_gemma2_text"

    vocab_size: int = 262_144
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 256
    max_position_embeddings: int = 1024
    sliding_window: int = 1024
    hidden_size_per_layer_input: int = 512
    use_bidirectional_attention: Literal["all", "vision"] | None = "all"
    embedding_dim: int = 768
    # There is no language modeling head, so there is nothing to tie the input embeddings to.
    tie_word_embeddings: bool = False

    # EmbeddingGemma 2 uses projection-only PLE: there is no per-layer token lookup table.
    vocab_size_per_layer_input = AttributeError()
    # There is no language modeling head, so logit softcapping is never applied.
    final_logit_softcapping = AttributeError()

    def __post_init__(self, **kwargs):
        # Per-layer attention shapes come from the checkpoint, never from a constant baked in here.
        # An explicit `per_layer_config` (which `convert_embedding_gemma2_weights.py` passes, giving
        # the global layers `head_dim=512` and a single KV head) is used as-is. Without one, every
        # layer falls back to the model-level `head_dim`/`num_key_value_heads`, whereas Gemma 4 would
        # otherwise impose its own `global_head_dim` of 512 on the full-attention layers. The KV-head
        # half of Gemma 4's override is already inert here, since it is gated on `attention_k_eq_v`.
        kwargs.setdefault("global_head_dim", self.head_dim)
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="google/embeddinggemma-2")
class EmbeddingGemma2Config(Gemma4Config):
    r"""
    text_config (`EmbeddingGemma2TextConfig`, *optional*):
        Configuration of the text backbone.
    vision_config (`Gemma4VisionConfig`, *optional*):
        Configuration of the vision tower. Reused verbatim from Gemma 4; the tower itself is
        resolved at runtime through `AutoModel`.
    audio_config (`Gemma4AudioConfig`, *optional*):
        Configuration of the audio tower. Reused verbatim from Gemma 4; the tower itself is
        resolved at runtime through `AutoModel`.
    boi_token_id (`int`, *optional*, defaults to 255999):
        The begin-of-image token index to wrap the image prompt.
    eoi_token_id (`int`, *optional*, defaults to 258882):
        The end-of-image token index to wrap the image prompt.
    boa_token_id (`int`, *optional*, defaults to 256000):
        The begin-of-audio token index to wrap the audio prompt.
    eoa_token_index (`int`, *optional*, defaults to 258883):
        The end-of-audio token index to wrap the audio prompt.
    """

    model_type = "embedding_gemma2"
    sub_configs = {
        "text_config": EmbeddingGemma2TextConfig,
        "vision_config": Gemma4VisionConfig,
        "audio_config": Gemma4AudioConfig,
    }

    text_config: EmbeddingGemma2TextConfig | dict[str, Any] | None = None
    vision_config: Gemma4VisionConfig | dict[str, Any] | None = None
    audio_config: Gemma4AudioConfig | dict[str, Any] | None = None
    # There is no language modeling head, so there is nothing to tie the input embeddings to.
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = EmbeddingGemma2TextConfig()
            logger.info("text_config is None. Using default EmbeddingGemma2TextConfig.")
        elif isinstance(self.text_config, dict):
            self.text_config = EmbeddingGemma2TextConfig(**self.text_config)

        if self.vision_config is None:
            logger.info("vision_config is None. EmbeddingGemma2Model.vision_tower will not be initialized.")
        if isinstance(self.vision_config, dict):
            self.vision_config = Gemma4VisionConfig(**self.vision_config)

        if self.audio_config is None:
            logger.info("audio_config is None. EmbeddingGemma2Model.audio_tower will not be initialized.")
        if isinstance(self.audio_config, dict):
            self.audio_config = Gemma4AudioConfig(**self.audio_config)

        PreTrainedConfig.__post_init__(self, **kwargs)


class EmbeddingGemma2RMSNorm(Gemma4RMSNorm):
    pass


class EmbeddingGemma2TextRotaryEmbedding(Gemma4TextRotaryEmbedding):
    pass


class EmbeddingGemma2TextScaledWordEmbedding(Gemma4TextScaledWordEmbedding):
    pass


class EmbeddingGemma2TextRouter(Gemma4TextRouter):
    pass


class EmbeddingGemma2TextExperts(Gemma4TextExperts):
    pass


class EmbeddingGemma2TextDecoderLayer(Gemma4TextDecoderLayer):
    # The signature is narrowed to the text config: Gemma 4 also allows a vision config here (its
    # vision encoder reuses this layer), but EmbeddingGemma 2 only ever builds text layers. The body
    # is inherited verbatim.
    def __init__(self, config: EmbeddingGemma2TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)


class EmbeddingGemma2PreTrainedModel(Gemma4PreTrainedModel):
    config: EmbeddingGemma2Config
    _no_split_modules = ["EmbeddingGemma2TextDecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        # Only text modules are initialized here. The vision and audio towers are separate
        # `PreTrainedModel`s resolved through `AutoModel`, so `smart_apply` dispatches to their
        # own `_init_weights` rather than this one.
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, EmbeddingGemma2TextRotaryEmbedding):
            for layer_type, rope_init_fn in module.rope_init_fns.items():
                rope_config = module.config.per_layer_config[layer_type]
                curr_inv_freq, _ = rope_init_fn(rope_config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, EmbeddingGemma2TextScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        elif isinstance(module, EmbeddingGemma2TextRouter):
            init.ones_(module.scale)
            init.ones_(module.per_expert_scale)
        elif isinstance(module, EmbeddingGemma2TextExperts):
            std = self.config.initializer_range
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, EmbeddingGemma2TextDecoderLayer):
            init.ones_(module.layer_scalar)

    def get_per_layer_input_embeddings(self):
        raise AttributeError("EmbeddingGemma 2 uses projection-only PLE and has no per-layer embedding table.")

    # Not a setter: `raise AttributeError` instructs the modular converter to delete the inherited
    # method, so it does not exist in the generated file at all.
    # trf-ignore: TRF033
    def set_per_layer_input_embeddings(self, value):
        raise AttributeError("EmbeddingGemma 2 uses projection-only PLE and has no per-layer embedding table.")

    def resize_token_embeddings(self):
        raise AttributeError("EmbeddingGemma 2 uses projection-only PLE and has no per-layer embedding table.")

    def _resize_per_layer_embeddings(self):
        raise AttributeError("EmbeddingGemma 2 uses projection-only PLE and has no per-layer embedding table.")


@auto_docstring
@dataclass
class EmbeddingGemma2TextModelOutputWithPast(Gemma4TextModelOutputWithPast):
    pass


@auto_docstring(
    custom_intro="""
    The EmbeddingGemma 2 text backbone. Unlike Gemma 4 it owns the `embedding_projection` that maps the final
    hidden states down to `config.embedding_dim`, so that the composite model needs no `forward` override.
    """
)
class EmbeddingGemma2TextModel(Gemma4TextModel):
    config: EmbeddingGemma2TextConfig

    def __init__(self, config: EmbeddingGemma2TextConfig):
        # Explicit parent call: the converter rewrites this to `super().__init__(config)`. Calling
        # `super()` directly here would instead splice in the whole of `Gemma4TextModel.__init__`,
        # reintroducing the `embed_tokens_per_layer` lookup table we do not have.
        PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = EmbeddingGemma2TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [EmbeddingGemma2TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = EmbeddingGemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = EmbeddingGemma2TextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.unique_layer_types = set(self.config.layer_types)

        # Projection-only Per-Layer Embeddings: the per-layer residual signal is derived entirely
        # from `inputs_embeds`. There is no `embed_tokens_per_layer` lookup table, so there is also
        # no token-identity term to blend in (and hence no `per_layer_input_scale`).
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_model_projection_scale = config.hidden_size**-0.5
        self.per_layer_projection_norm = EmbeddingGemma2RMSNorm(
            config.hidden_size_per_layer_input, eps=config.rms_norm_eps
        )

        # The embedding head. Applying it per-token here is equivalent to applying it after the
        # mean pooling that SentenceTransformers performs downstream, since a linear map commutes
        # with averaging.
        self.embedding_projection = nn.Linear(config.hidden_size, config.embedding_dim, bias=False)

        # Update `_keys_to_ignore_on_load_unexpected` to drop all k/v proj and norms for the shared layers
        self._keys_to_ignore_on_load_unexpected = []
        for i, layer in enumerate(self.layers):
            if layer.self_attn.is_kv_shared_layer:
                self._keys_to_ignore_on_load_unexpected.extend(
                    [f"layers.{i}.self_attn.{name}" for name in ("k_proj", "v_proj", "k_norm", "v_norm")]
                )

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> EmbeddingGemma2TextModelOutputWithPast:
        r"""
        per_layer_inputs (`torch.Tensor`, *optional*):
            Unused by EmbeddingGemma 2, which computes its per-layer inputs entirely from
            `inputs_embeds`. Accepted for signature compatibility with Gemma 4.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        shared_kv_states = kwargs.pop("shared_kv_states", UserDict())

        # decoder layers
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            per_layer_input = per_layer_inputs[:, :, i, :]

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.embedding_projection(hidden_states)

        return EmbeddingGemma2TextModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            shared_kv_states=shared_kv_states if kwargs.get("return_shared_kv_states", False) else None,
        )

    def get_per_layer_inputs(self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None) -> None:
        """EmbeddingGemma 2 has no per-layer embedding table, so there is no token-identity term.

        Returning `None` (rather than raising) lets the composite `EmbeddingGemma2Model.forward`
        reuse Gemma 4's multimodal path unchanged: it simply leaves `per_layer_inputs` unset and
        the projection-only signal is computed in `project_per_layer_inputs`.
        """
        return None

    def project_per_layer_inputs(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Compute the per-layer residual signal from `inputs_embeds` alone."""
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return self.per_layer_projection_norm(per_layer_projection)


class EmbeddingGemma2MultimodalEmbedder(Gemma4MultimodalEmbedder):
    # The body is inherited verbatim; only the annotation changes. Gemma 4's signature names its own
    # vision/audio config classes, which the converter would rename to classes we do not generate --
    # EmbeddingGemma 2 reuses the Gemma 4 ones as-is so that `AutoModel` resolves the towers.
    def __init__(
        self,
        multimodal_config: Gemma4AudioConfig | Gemma4VisionConfig,
        text_config: EmbeddingGemma2TextConfig,
    ):
        super().__init__(multimodal_config, text_config)


@auto_docstring(
    custom_intro="""
    The EmbeddingGemma 2 model: a vision backbone, an audio backbone and a text backbone whose final hidden
    states are projected to `config.text_config.embedding_dim`. Intended to be wrapped by SentenceTransformers'
    mean pooling and normalization.
    """
)
class EmbeddingGemma2Model(Gemma4Model):
    config: EmbeddingGemma2Config

    def __init__(self, config: EmbeddingGemma2Config):
        # Explicit parent call so the converter emits `super().__init__(config)` rather than splicing
        # in `Gemma4Model.__init__`, which reads `config.text_config.vocab_size_per_layer_input` --
        # an attribute EmbeddingGemma 2 deletes.
        PreTrainedModel.__init__(self, config)
        # CODEPATH: `vision_config` is set on the released multimodal EmbeddingGemma 2 checkpoints;
        # it is None only for a text-only config built by hand.
        self.vision_tower = AutoModel.from_config(config.vision_config) if config.vision_config is not None else None
        self.vocab_size = config.text_config.vocab_size

        language_model = AutoModel.from_config(config=config.text_config)
        self.language_model = language_model
        # CODEPATH: `audio_config` is set on the released multimodal EmbeddingGemma 2 checkpoints;
        # it is None only for a text-only config built by hand.
        self.audio_tower = AutoModel.from_config(config.audio_config) if config.audio_config is not None else None
        self.embed_vision = (
            # CODEPATH: paired with `vision_tower` above -- present on the released multimodal
            # checkpoints, None for a hand-built text-only config.
            EmbeddingGemma2MultimodalEmbedder(config.vision_config, config.text_config)
            if config.vision_config is not None
            else None
        )
        self.embed_audio = (
            # CODEPATH: paired with `audio_tower` above -- present on the released multimodal
            # checkpoints, None for a hand-built text-only config.
            EmbeddingGemma2MultimodalEmbedder(config.audio_config, config.text_config)
            if config.audio_config is not None
            else None
        )
        self.post_init()

    def get_per_layer_input_embeddings(self):
        raise AttributeError("EmbeddingGemma 2 uses projection-only PLE and has no per-layer embedding table.")

    # Not a setter: `raise AttributeError` instructs the modular converter to delete the inherited
    # method, so it does not exist in the generated file at all.
    # trf-ignore: TRF033
    def set_per_layer_input_embeddings(self, value):
        raise AttributeError("EmbeddingGemma 2 uses projection-only PLE and has no per-layer embedding table.")


class EmbeddingGemma2VideoProcessorKwargs(Gemma4VideoProcessorKwargs):
    """
    patch_size (`int`, *optional*):
        Size of each image patch in pixels.
    max_soft_tokens (`int`, *optional*):
        Maximum number of soft (vision) tokens per video frame.
        Must be one of {70, 140, 280, 560, 1120}.
    pooling_kernel_size (`int`, *optional*):
        Spatial pooling kernel size applied after patchification.
    exclude_timestamps (`bool`, *optional*):
        Whether to exclude frame timestamps from the video placeholder expansion.
    use_1fps_linear_sampling (`bool`, *optional*):
        Whether to sample frames using 1-FPS linspace sequence sampling matching internal Google3 pipelines.
    """

    exclude_timestamps: bool
    use_1fps_linear_sampling: bool


class EmbeddingGemma2VideoProcessor(Gemma4VideoProcessor):
    # EmbeddingGemma 2 was trained on visual-only, 1-FPS-sampled video, so both default to `True`.
    use_1fps_linear_sampling = True
    exclude_timestamps = True
    valid_kwargs = EmbeddingGemma2VideoProcessorKwargs

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        use_1fps_linear_sampling: bool | None = None,
        **kwargs,
    ) -> np.ndarray:
        use_1fps_linear_sampling = (
            use_1fps_linear_sampling if use_1fps_linear_sampling is not None else self.use_1fps_linear_sampling
        )

        if use_1fps_linear_sampling:
            num_frames = num_frames if num_frames is not None else self.num_frames
            total_num_frames = metadata.total_num_frames if metadata is not None else None

            if (
                total_num_frames is None
                and metadata is not None
                and metadata.duration is not None
                and metadata.fps is not None
            ):
                total_num_frames = int(metadata.duration * metadata.fps)

            if metadata is None or metadata.fps is None or total_num_frames is None:
                # Pre-extracted frames without FPS metadata: keep all frames if <= num_frames, else uniformly sample
                if total_num_frames is not None and total_num_frames <= num_frames:
                    return np.arange(total_num_frames)
                if total_num_frames is not None:
                    return np.linspace(0, total_num_frames - 1, num_frames, dtype=int)
                raise ValueError(
                    "Asked to sample with `use_1fps_linear_sampling=True`, but no video metadata was provided or "
                    "`total_num_frames` is missing. Please pass in `VideoMetadata` object with total_num_frames."
                )

            video_fps = metadata.fps
            total_seconds = max(1, int(total_num_frames / video_fps))
            sec_indices = [min(total_num_frames - 1, int(s * video_fps)) for s in range(total_seconds)]
            if len(sec_indices) <= num_frames:
                return np.array(sec_indices)
            linspace_idx = np.linspace(0, len(sec_indices) - 1, num_frames, dtype=int)
            return np.array([sec_indices[i] for i in linspace_idx])

        # Explicit parent call: rewritten to `super().sample_frames(...)` by the converter.
        # `Gemma4VideoProcessor` does not define `sample_frames`, so a literal `super()` call
        # here would make the converter fail looking for a parent body to splice.
        return BaseVideoProcessor.sample_frames(self, metadata, num_frames=num_frames, fps=fps, **kwargs)


class EmbeddingGemma2ProcessorKwargs(Gemma4ProcessorKwargs):
    # EmbeddingGemma 2 has no image processor of its own -- `Gemma4ImageProcessor` is reused via the
    # auto mapping -- so the kwargs typed dict is reused too.
    images_kwargs: Gemma4ImageProcessorKwargs
    videos_kwargs: EmbeddingGemma2VideoProcessorKwargs


class EmbeddingGemma2Processor(Gemma4Processor):
    valid_processor_kwargs = EmbeddingGemma2ProcessorKwargs

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs,
    ):
        images, text, videos, audio = ProcessorMixin.prepare_inputs_layout(
            self, images=images, text=text, videos=videos, audio=audio, **kwargs
        )

        # Model requires nested struct
        if images is not None:
            images = make_nested_list_of_images(images)

        # Normalize videos so len(videos) gives the number of videos, not frames
        if videos is not None:
            videos = make_batched_videos(videos)

        # Embedding inputs are frequently a bare image / video / audio with no text at all,
        # so synthesize the placeholder text for every modality.
        if images and not text:
            text = [" ".join([self.image_token] * len(image_list)) for image_list in images]
        if audio and not text:
            text = [self.audio_token] * len(audio)
        if videos is not None and not text:
            text = [self.video_token] * len(videos)

        return images, text, videos, audio

    def validate_inputs(
        self,
        images: ImageInput | list[ImageInput] | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        ProcessorMixin.validate_inputs(self, images=images, text=text, **kwargs)

        # Unlike Gemma 4, any single modality on its own is a valid embedding input.
        if text is None and images is None and videos is None and audio is None:
            raise ValueError("You must provide at least one of `text`, `images`, `videos`, or `audio`.")

        if audio is not None and (self.audio_token is None or self.boa_token is None or self.eoa_token is None):
            raise ValueError("Audio inputs were provided, but the tokenizer does not have an `audio_token` defined.")

        if text is not None:
            n_images_in_text = [sample.count(self.image_token) for sample in text]
            if images is not None:
                if len(images) != len(text):
                    raise ValueError(
                        f"Received inconsistently sized batches of images ({len(images)}) and text ({len(text)})."
                    )

                n_images_in_images = [len(sublist) for sublist in images]
                if n_images_in_text != n_images_in_images:
                    raise ValueError(
                        f"The total number of {self.image_token} tokens in the prompts should be the same as the number of images passed."
                        f" Found {n_images_in_text} {self.image_token} tokens and {n_images_in_images} images per sample."
                    )
            elif images is None and any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        num_soft_tokens = video_inputs["num_soft_tokens_per_video"][video_idx]
        exclude_timestamps = kwargs.get("exclude_timestamps", self.video_processor.exclude_timestamps)

        # Visual-only mode: matches the EmbeddingGemma 2 training distribution
        if exclude_timestamps:
            num_frames = video_inputs["pixel_values_videos"][video_idx].shape[0]
            frame_str = f"{self.boi_token}{self.video_token * num_soft_tokens}{self.eoi_token}"
            return "".join([frame_str] * num_frames)

        metadata = video_inputs["video_metadata"][video_idx]

        if metadata.fps is None:
            logger.warning_once(
                "EmbeddingGemma 2 requires frame timestamps to construct prompts, but the `fps` of the input video "
                "could not be inferred. Probably `video_metadata` was missing from inputs and you passed pre-sampled "
                "frames. Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
            )
        metadata.fps = 24 if metadata.fps is None else metadata.fps

        # mm:ss format for timestamps
        timestamp_str = [f"{int(seconds // 60):02d}:{int(seconds % 60):02d}" for seconds in metadata.timestamps]
        return " ".join(
            [f"{t} {self.boi_token}{self.video_token * num_soft_tokens}{self.eoi_token}" for t in timestamp_str]
        )


__all__ = [
    "EmbeddingGemma2Config",
    "EmbeddingGemma2Model",
    "EmbeddingGemma2PreTrainedModel",
    "EmbeddingGemma2Processor",
    "EmbeddingGemma2TextConfig",
    "EmbeddingGemma2TextModel",
    "EmbeddingGemma2VideoProcessor",
]
