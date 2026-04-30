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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessorMixin, Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    torch_compilable_check,
)
from ...utils.output_capturing import capture_outputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding
from ..llava.modeling_llava import LlavaCausalLMOutputWithPast, LlavaForConditionalGeneration, LlavaModelOutputWithPast
from ..qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLPreTrainedModel,
    Qwen2VLVisionBlock,
    VisionAttention,
    VisionMlp,
)
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor


class Kimi2_6VisionConfig(PreTrainedConfig):
    r"""
    pos_emb_height (`int`, *optional*):
        Initial position embedding height.
    pos_emb_width (`int`, *optional*):
        Initial position embedding width.
    pos_emb_time (`int`, *optional*):
        Initial position embedding time dimension.
    merge_kernel_size (`tuple[int] | list[int]`, *optional*):
        Kernel size for patch merging.
    """

    model_type = "kimi2_6_vision"

    patch_size: int = 14
    pos_emb_height: int = 64
    pos_emb_width: int = 64
    pos_emb_time: int = 4
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    hidden_act: str = "gelu_pytorch_tanh"
    merge_kernel_size: tuple[int, int] | list[int] = (2, 2)
    rope_parameters: dict | None = None
    max_position_embeddings: int | None = None


class Kimi2_6Config(PreTrainedConfig):
    r"""
    projection_ln_eps (`float`, *optional*):
        Layer norm epsilon for projector.
    """

    model_type = "kimi2_6"
    sub_configs = {"text_config": AutoConfig, "vision_config": Kimi2_6VisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_hidden_size: int | None = 1152
    projection_hidden_act: str = "gelu"
    projection_ln_eps: float = 1e-5
    image_token_id: int = 163605
    video_token_id: int = 163606
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "deepseek_v3")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["deepseek_v3"]()

        if isinstance(self.vision_config, dict):
            self.vision_config = Kimi2_6VisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Kimi2_6VisionConfig()
        super().__post_init__(**kwargs)


class Kimi2_6ModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class Kimi2_6CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class Kimi2_6VisionPositionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_frames = config.pos_emb_time

        self.position_embeddings = nn.Parameter(
            torch.empty(config.pos_emb_height, config.pos_emb_width, config.hidden_size)
        )
        time_position_embeddings = self.get_1d_sincos_pos_embed()
        self.register_buffer("time_position_embeddings", time_position_embeddings, persistent=False)

    # TODO: compute in torch
    def get_1d_sincos_pos_embed(self):
        grid_t = np.arange(self.num_frames, dtype=np.float32)
        omega = np.arange(self.dim // 2, dtype=np.float32)
        omega /= self.dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        grid_t = grid_t.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", grid_t, omega)  # (M, D/2), outer product
        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        pos_embed = torch.tensor(pos_embed, dtype=torch.float).unsqueeze(1)
        return pos_embed

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thw.tolist():
            if t > self.num_frames:
                raise ValueError(
                    f"Got an input with {t} frames. Number of frames should be less than config.pos_emb_time=({self.num_frames})"
                )

            if (h, w) == self.position_embeddings.shape[:-1]:
                position_embeddings = self.position_embeddings.flatten(0, 1)
            else:
                position_embeddings = self.position_embeddings.permute(2, 0, 1).unsqueeze(0)
                position_embeddings = F.interpolate(
                    position_embeddings,
                    size=(h, w),
                    mode="bicubic",
                )
                position_embeddings = position_embeddings.squeeze(0).permute(1, 2, 0).flatten(0, 1)

            position_embeddings = position_embeddings.unsqueeze(0).repeat(t, 1, 1)
            if t > 1:
                position_embeddings = position_embeddings + self.time_position_embeddings[0:t]

            pos_embs.append(position_embeddings.flatten(0, 1))
        hidden_states = hidden_states + torch.cat(pos_embs)
        return hidden_states


class Kimi2_6VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_size = (
            config.patch_size if not isinstance(config.patch_size, int) else (config.patch_size, config.patch_size)
        )
        self.proj = nn.Conv2d(3, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = Kimi2_6VisionPositionEmbeddings(config)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(pixel_values).view(pixel_values.size(0), -1)
        hidden_states = self.pos_emb(hidden_states, grid_thw)
        return hidden_states


class Kimi2_6VisionRotaryEmbeddings(Gemma4VisionRotaryEmbedding):
    pass


class Kimi2_6VisionMLP(VisionMlp):
    pass


class Kimi2_6VisionAttention(VisionAttention):
    def __init__(self, config: Kimi2_6VisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads


class Kimi2_6VisionEncoderLayer(Qwen2VLVisionBlock):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.intermediate_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.intermediate_size, eps=1e-6)

        self.attn = Kimi2_6VisionAttention(config=config)
        self.mlp = Kimi2_6VisionMLP(config.intermediate_size, config.hidden_size, config.hidden_act)


class Kimi2_6PreTrainedModel(Qwen2VLPreTrainedModel):
    _no_split_modules = ["Kimi2_6VisionEncoderLayer"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)


class Kimi2_6VisionModel(Kimi2_6PreTrainedModel):
    config: Kimi2_6VisionConfig
    input_modalities = ("image", "video")
    can_record_outputs = {
        "hidden_states": Kimi2_6VisionEncoderLayer,
        "attentions": Kimi2_6VisionAttention,
    }

    def __init__(self, config: Kimi2_6VisionConfig):
        super().__init__(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_embed = Kimi2_6VisionPatchEmbed(config)

        self.rotary_emb = Kimi2_6VisionRotaryEmbeddings(config)
        self.encoder_blocks = nn.ModuleList(
            [Kimi2_6VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_init()

    def temporal_patch_merger(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        hidden_dim = hidden_states.size(-1)
        kernel_height, kernel_width = self.merge_kernel_size

        outputs = []
        pre_sum = 0
        for t, h, w in grid_thw.tolist():
            # Get the current sequence
            seq = hidden_states[pre_sum : pre_sum + t * h * w]
            # Reshape along self.merge_kernel_size and concat to the last dimension
            new_height, new_width = h // kernel_height, w // kernel_width
            reshaped_seq = seq.view(t, new_height, kernel_height, new_width, kernel_width, hidden_dim)
            reshaped_seq = reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)  # temporal pooling
            padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
            outputs.append(padded_seq)
            pre_sum += t * h * w

        return torch.cat(outputs, dim=0)

    def get_position_ids(self, grid_thw: torch.Tensor) -> torch.Tensor:
        "Builds (h_pos, w_pos) grid for each sample, then cat across batch"
        all_position_ids = []
        for t, h, w in grid_thw.tolist():
            h_ids = torch.arange(h, device=grid_thw.device)
            w_ids = torch.arange(w, device=grid_thw.device)

            # (h, w, 2) grid of (row, col) coordinates
            grid = torch.stack(torch.meshgrid(h_ids, w_ids, indexing="ij"), dim=-1)

            # (h*w, 2) -> repeat for each temporal frame -> (t*h*w, 2)
            all_position_ids.append(grid.reshape(-1, 2).repeat(t, 1))

        position_ids = torch.cat(all_position_ids, dim=0).unsqueeze(0)
        return position_ids  # (1, total_patches, 2)

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        hidden_states = self.patch_embed(pixel_values, grid_thw=grid_thw)
        position_ids = self.get_position_ids(grid_thw=grid_thw)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thw.dtype, device=grid_thw.device),
                grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2],
            )
        )

        max_seqlen = lengths.max()
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for block in self.encoder_blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.final_layernorm(hidden_states)
        pooled_hidden_states = self.temporal_patch_merger(hidden_states, grid_thw)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_hidden_states,
        )


class Kimi2_6MultimodalProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.vision_config.hidden_size * (
            config.vision_config.merge_kernel_size[0] * config.vision_config.merge_kernel_size[1]
        )
        self.pre_norm = nn.LayerNorm(config.projection_hidden_size, eps=config.projection_ln_eps)

        self.in_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        batch_size = hidden_states.shape[0]
        hidden_states = self.pre_norm(hidden_states).view(batch_size, -1, self.hidden_size)
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class Kimi2_6Model(Kimi2_6PreTrainedModel):
    def __init__(self, config: Kimi2_6Config):
        super().__init__(config)
        self.vision_tower = Kimi2_6VisionModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.mm_projector = Kimi2_6MultimodalProjection(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        vision_outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw, **kwargs)
        image_embeds = self.mm_projector(vision_outputs.pooler_output)
        vision_outputs.pooler_output = image_embeds
        return vision_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
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
        if image_features is not None:
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )
        return special_image_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Kimi2_6ModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return Kimi2_6ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Kimi2_6ForConditionalGeneration(LlavaForConditionalGeneration):
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width of feature shape of each image in LLM.
        """
        return self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Kimi2_6CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Kimi2_6ForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("TODO")
        >>> processor = AutoProcessor.from_pretrained("TODO")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        >>> output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """

        outputs: Kimi2_6ModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
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

        return Kimi2_6CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Kimi2_6Processor(Qwen2VLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        ProcessorMixin.__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_token = "<|media_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|media_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )


__all__ = [
    "Kimi2_6Config",
    "Kimi2_6VisionConfig",
    "Kimi2_6ForConditionalGeneration",
    "Kimi2_6Model",
    "Kimi2_6PreTrainedModel",
    "Kimi2_6VisionModel",
    "Kimi2_6Processor",
]
