import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils_fast import (
    DefaultFastImageProcessorKwargs,
)
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    SizeDict,
    get_image_size,
)
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
    Qwen2VLFastImageProcessorKwargs,
    Qwen2VLImageProcessorFast,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
    TransformersKwargs,
    VisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.models.qwen2_vl.processing_qwen2_vl import (
    Qwen2VLImagesKwargs,
    Qwen2VLProcessor,
    Qwen2VLProcessorKwargs,
)
from transformers.models.qwen2_vl.video_processing_qwen2_vl import (
    Qwen2VLVideoProcessor,
    Qwen2VLVideoProcessorInitKwargs,
)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipMLP,
)
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import (
    TensorType,
    auto_docstring,
    can_return_tuple,
    logging,
)
from transformers.video_utils import (
    VideoInput,
    group_videos_by_shape,
    make_batched_videos,
    reorder_videos,
)

from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Videollama3VisionConfig(SiglipVisionConfig):
    """
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "videollama3_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            **kwargs,
        )

        self.initializer_range = initializer_range
        del self.image_size


class Videollama3Config(PretrainedConfig):
    """
    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Videollama3VisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to -1):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to -1):
            The video token index to encode the image prompt.
    """

    model_type = "videollama3"
    sub_configs = {"vision_config": Videollama3VisionConfig, "text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            raise ValueError(
                f"vision_config must be of type `dict` or `PretrainedConfig`, but got {type(vision_config)}."
            )

        if isinstance(text_config, dict):
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()
        else:
            raise ValueError(f"text_config must be of type `dict` or `PretrainedConfig`, but got {type(text_config)}.")

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(**kwargs)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for VideoLLaMA3 outputs, with hidden states and attentions.
    """
)
class Videollama3ModelOutputWithPast(ModelOutput):
    """
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_images_features, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_features, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    video_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for VideoLLaMA3 causal language model (or autoregressive) outputs.
    """
)
class Videollama3CausalLMOutputWithPast(ModelOutput):
    """
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_images_features, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_features, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    video_hidden_states: Optional[torch.FloatTensor] = None


class Videollama3VisionRotaryEmbedding(VisionRotaryEmbedding):
    pass


class Videollama3VisionEmbeddings(nn.Module):
    def __init__(self, config: Videollama3VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.config.num_channels, self.patch_size, self.patch_size)
        patch_embeds = self.patch_embedding(hidden_states)
        embeddings = patch_embeds.view(-1, self.embed_dim)
        return embeddings


class Videollama3VisionMLP(SiglipMLP):
    pass


class Videollama3VisionAttention(SiglipAttention):
    def __init__(self, config):
        super().__init__(config)
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        del self.scale
        del self.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            cu_seqlens (`torch.Tensor` of shape `(num_images_or_videos + 1,)`):
                The cumulative sequence lengths of each image or video feature.
            position_embeddings (`tuple(torch.Tensor, torch.Tensor)` of shape `(num_patches, head_dim // 2)`):
                The cosine and sine position embeddings for vision attention.
        """
        seq_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs, attn_weights = [], []
            for q, k, v in zip(*splits):
                attn_output, attn_weight = attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )
                attn_outputs.append(attn_output)
                attn_weights.append(attn_weight)

            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Videollama3VisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: Videollama3VisionConfig):
        super().__init__(config)
        self.self_attn = Videollama3VisionAttention(config=config)
        self.mlp = Videollama3VisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            cu_seqlens (`torch.Tensor` of shape `(num_images_or_videos + 1,)`):
                The cumulative sequence lengths of each image or video feature.
            position_embeddings (`tuple(torch.Tensor, torch.Tensor)` of shape `(num_patches, head_dim // 2)`):
                The cosine and sine position embeddings for vision attention.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Videollama3VisionEncoder(SiglipEncoder):
    def __init__(self, config: Videollama3VisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Videollama3VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutput]:
        """
        cu_seqlens (`torch.Tensor` of shape `(num_images_or_videos + 1,)`):
            The cumulative sequence lengths of each image or video feature.
        position_embeddings (`tuple(torch.Tensor, torch.Tensor)` of shape `(num_patches, head_dim // 2)`):
            The cosine and sine position embeddings for vision attention.
        """
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutput(last_hidden_state=hidden_states)


class Videollama3PreTrainedModel(Qwen2VLPreTrainedModel):
    config: Videollama3Config
    _no_split_modules = ["Qwen2DecoderLayer", "Videollama3VisionEncoderLayer"]


class Videollama3VisionModel(Videollama3PreTrainedModel):
    config: Videollama3VisionConfig
    _no_split_modules = ["Videollama3VisionEncoderLayer"]
    _can_record_outputs = {
        "hidden_states": Videollama3VisionEncoderLayer,
        "attentions": Videollama3VisionAttention,
    }

    def __init__(self, config: Videollama3VisionConfig):
        super().__init__(config)
        head_dim = config.hidden_size // config.num_attention_heads

        self.rotary_pos_emb = Videollama3VisionRotaryEmbedding(head_dim // 2)
        self.embeddings = Videollama3VisionEmbeddings(config)
        self.encoder = Videollama3VisionEncoder(config)
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    def rot_pos_emb(self, grid_thw, merge_sizes):
        pos_ids = []
        for (t, h, w), merge_size in zip(grid_thw, merge_sizes):
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_thw = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_thw)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)

        return rotary_pos_emb

    def pixel_unshuffle(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        merge_sizes: torch.Tensor,
    ):
        hidden_states_chunks = hidden_states.split(grid_thw.prod(dim=1).tolist(), dim=0)
        outputs = []

        for hidden_states, (t, h, w), merge_size in zip(hidden_states_chunks, grid_thw, merge_sizes):
            c = hidden_states.shape[-1]
            hidden_states = hidden_states.view(t, h // merge_size, w // merge_size, merge_size, merge_size, c).permute(
                0, 1, 3, 2, 4, 5
            )
            hidden_states = hidden_states.reshape(t, h, w, c).permute(0, 3, 1, 2)
            hidden_states = torch.nn.functional.interpolate(
                hidden_states, size=(h // merge_size, w // merge_size), mode="bilinear"
            )
            hidden_states = hidden_states.permute(0, 2, 3, 1).view(-1, c)
            outputs.append(hidden_states)

        return torch.cat(outputs, dim=0)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        merge_sizes: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutput]:
        """
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width dimensions of feature shape for each image. Each row contains [t, h, w] values.
        merge_sizes (`torch.Tensor` of shape `(num_images_or_videos,)`):
            The spatial downsampling ratio of each image or video feature.
        """
        rotary_pos_emb = self.rot_pos_emb(grid_thw, merge_sizes)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = self.embeddings(pixel_values.type(self.dtype))
        encoder_outputs: BaseModelOutput = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        last_hidden_state = self.pixel_unshuffle(last_hidden_state, grid_thw, merge_sizes)

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class Videollama3Projector(nn.Module):
    def __init__(self, config: Videollama3Config) -> None:
        super().__init__()
        in_hidden_size = config.vision_config.hidden_size
        out_hidden_size = config.text_config.hidden_size
        self.readout = nn.Sequential(
            nn.Linear(in_hidden_size, out_hidden_size),
            nn.GELU(),
            nn.Linear(out_hidden_size, out_hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.readout(hidden_states)
        return hidden_states


class Videollama3Model(Qwen2VLModel, PreTrainedModel):
    _checkpoint_conversion_mapping = {}
    _no_split_modules = ["Qwen2DecoderLayer", "Videollama3VisionEncoderLayer"]
    _can_compile_fullgraph = False

    def __init__(self, config: Videollama3Config):
        PreTrainedModel.__init__(config)
        self.vision_model = AutoModel.from_config(config.vision_config)
        self.projector = Videollama3Projector(config)
        self.language_model = AutoModel.from_config(config.text_config)

        self.post_init()

    def get_rope_index(self):
        raise AttributeError("Not needed for VideoLLaMA3")

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        video_merge_sizes: torch.LongTensor,
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            video_merge_sizes (`torch.Tensor` of shape `(num_videos,)`):
                The spatial downsampling ratio of each video feature.
        """
        video_embeds = self.vision_model(
            pixel_values=pixel_values_videos,
            grid_thw=video_grid_thw,
            merge_sizes=video_merge_sizes,
            return_dict=True,
        ).last_hidden_state
        video_embeds = self.projector(video_embeds)

        split_sizes = video_grid_thw.prod(dim=1) // (video_merge_sizes**2)
        video_embeds = torch.split(video_embeds, split_sizes.tolist())

        return video_embeds

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        image_merge_sizes: torch.LongTensor,
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            image_merge_sizes (`torch.Tensor` of shape `(num_images,)`):
                The spatial downsampling ratio of each image feature.
        """
        image_embeds = self.vision_model(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
            merge_sizes=image_merge_sizes,
            return_dict=True,
        ).last_hidden_state
        image_embeds = self.projector(image_embeds)

        split_sizes = image_grid_thw.prod(dim=1) // (image_merge_sizes**2)
        image_embeds = torch.split(image_embeds, split_sizes.tolist())

        return image_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_merge_sizes: Optional[torch.LongTensor] = None,
        video_compression_mask: Optional[torch.BoolTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Videollama3ModelOutputWithPast]:
        """
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        image_merge_sizes (`torch.Tensor` of shape `(num_images,)`):
            The spatial downsampling ratio of each image feature.
        video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`):
            The temporal, height and width of feature shape of each video before vision encoder.
        video_merge_sizes (`torch.Tensor` of shape `(num_videos,)`):
            The spatial downsampling ratio of each video feature.
        video_compression_mask (`torch.BoolTensor` of shape `(num_video_features,)`, *optional*):
            The mask to indicate which video features are kept after token compression.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, image_merge_sizes)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        video_embeds = None
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, video_merge_sizes)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            if video_compression_mask is not None:
                video_embeds = video_embeds[video_compression_mask]
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Videollama3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_embeds,
            video_hidden_states=video_embeds,
        )
        return output if return_dict else output.to_tuple()


class Videollama3ForConditionalGeneration(Qwen2VLForConditionalGeneration, PreTrainedModel):
    _checkpoint_conversion_mapping = {}
    _no_split_modules = ["Qwen2DecoderLayer", "Videollama3VisionEncoderLayer"]
    _can_compile_fullgraph = False

    def __init__(self, config: Videollama3Config):
        PreTrainedModel.__init__(config)
        self.model = Videollama3Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def visual(self):
        raise AttributeError("Not needed for VideoLLaMA3")

    @property
    def vision_model(self):
        return self.model.vision_model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_merge_sizes: Optional[torch.LongTensor] = None,
        video_compression_mask: Optional[torch.BoolTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Videollama3CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        image_merge_sizes (`torch.Tensor` of shape `(num_images,)`):
            The spatial downsampling ratio of each image feature.
        video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`):
            The temporal, height and width of feature shape of each video before vision encoder.
        video_merge_sizes (`torch.Tensor` of shape `(num_videos,)`):
            The spatial downsampling ratio of each video feature.
        video_compression_mask (`torch.BoolTensor` of shape `(num_video_features,)`, *optional*):
            The mask to indicate which video features are kept after token compression.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_merge_sizes=image_merge_sizes,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            video_merge_sizes=video_merge_sizes,
            video_compression_mask=video_compression_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Videollama3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_merge_sizes: Optional[torch.LongTensor] = None,
        video_compression_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_merge_sizes=image_merge_sizes,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            video_merge_sizes=video_merge_sizes,
            video_compression_mask=video_compression_mask,
            use_cache=use_cache,
            **kwargs,
        )

        if model_inputs["cache_position"][0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_merge_sizes: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_merge_sizes: Optional[torch.LongTensor] = None,
        video_compression_mask: Optional[torch.BoolTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id

        if inputs_embeds is not None:
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        if image_grid_thw is not None:
            num_image_features = image_grid_thw.prod(dim=1) // (image_merge_sizes**2)
        else:
            num_image_features = []

        if video_grid_thw is not None:
            num_video_features = video_grid_thw.prod(dim=1) // (video_merge_sizes**2)
            if video_compression_mask is not None:
                num_video_features = video_compression_mask.split(num_video_features.tolist())
                num_video_features = [mask.sum() for mask in num_video_features]
        else:
            num_video_features = []

        image_nums, video_nums = [], []
        start_image_idx, start_video_idx = 0, 0

        for num_image_tokens, num_video_tokens in zip(image_mask.sum(dim=1), video_mask.sum(dim=1)):
            cu_num_features = 0
            image_idx = start_image_idx
            while image_idx < len(num_image_features) and cu_num_features < num_image_tokens:
                cu_num_features += num_image_features[image_idx]
                image_idx += 1
            assert cu_num_features == num_image_tokens, (
                "The number of image tokens does not match the number of image features."
            )
            image_nums.append(image_idx - start_image_idx)
            start_image_idx = image_idx

            cu_num_features = 0
            video_idx = start_video_idx
            while video_idx < len(num_video_features) and cu_num_features < num_video_tokens:
                cu_num_features += num_video_features[video_idx]
                video_idx += 1
            assert cu_num_features == num_video_tokens, (
                "The number of video tokens does not match the number of video features."
            )
            video_nums.append(video_idx - start_video_idx)
            start_video_idx = video_idx

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "image_merge_sizes",
            "pixel_values_videos",
            "video_grid_thw",
            "video_merge_sizes",
            "video_compression_mask",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            video_merge_sizes = model_kwargs.get("video_merge_sizes", None)
            video_compression_mask = model_kwargs.get("video_compression_mask", None)

            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids,
                inputs_embeds=model_kwargs.get("inputs_embeds", None),
                image_grid_thw=image_grid_thw,
                image_merge_sizes=model_kwargs.get("image_merge_sizes", None),
                video_grid_thw=video_grid_thw,
                video_merge_sizes=video_merge_sizes,
                video_compression_mask=video_compression_mask,
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_merge_sizes":
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_compression_mask":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    merge_sizes = torch.split(video_merge_sizes, list(video_nums))
                    lengths = [
                        (torch.prod(sample, dim=1) // merge_size**2).sum()
                        for sample, merge_size in zip(samples, merge_sizes)
                    ]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_merge_sizes":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )

            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


class Videollama3ImagesKwargs(Qwen2VLImagesKwargs):
    pass


class Videollama3ProcessorKwargs(Qwen2VLProcessorKwargs):
    image_kwargs: Videollama3ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class Videollama3Processor(Qwen2VLProcessor):
    r"""
    Constructs a VideoLLaMA3 processor which wraps a VideoLLaMA3 image processor and a Qwen2 tokenizer into a single processor.
    [`Videollama3Processor`] offers all the functionalities of [`Videollama3ImageProcessor`] and [`Qwen2Tokenizer`]. See the
    [`~Videollama3Processor.__call__`] and [`~Videollama3Processor.decode`] for more information.
    Args:
        image_processor ([`Videollama3ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2Tokenizer`], *optional*):
            The tokenizer is a required input.
        video_processor ([`Videollama3VideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
    """

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Videollama3ProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Videollama3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
            image_merge_sizes = image_inputs["image_merge_sizes"]
        else:
            image_grid_thw = image_merge_sizes = []

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            num_video_tokens = [
                grid_thw.prod() // merge_size**2
                for grid_thw, merge_size in zip(videos_inputs["video_grid_thw"], videos_inputs["video_merge_sizes"])
            ]
            video_compression_masks = videos_inputs["video_compression_mask"].split(num_video_tokens)
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
            timestamps = []
            for metadata in video_metadata:
                if metadata.fps is None:
                    logger.warning_once(
                        "VideoLLaMA4 requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                        "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                        "Defaulting to `fps=1`. Please provide `video_metadata` for more accurate results."
                    )
                metadata.fps = 1 if metadata.fps is None else metadata.fps
                timestamps.append(metadata.timestamps)
        else:
            video_compression_masks = timestamps = []

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if images is not None:
            image_index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[image_index].prod() // (image_merge_sizes[image_index] ** 2)
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    image_index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            video_index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    frame_compression_masks = video_compression_masks[video_index].split(
                        len(video_compression_masks[video_index]) // len(timestamps[video_index])
                    )
                    num_frame_tokens = [x.sum() for x in frame_compression_masks]
                    frame_prompts = [
                        f"Time {t:.1f}s:" + "<|placeholder|>" * n
                        for n, t in zip(num_frame_tokens, timestamps[video_index])
                    ]
                    text[i] = text[i].replace(self.video_token, ",".join(frame_prompts), 1)
                    video_index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)


class Videollama3FastImageProcessorKwargs(Qwen2VLFastImageProcessorKwargs):
    pass


class Videollama3ImageProcessorFast(Qwen2VLImageProcessorFast):
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    temporal_patch_size = 1
    merge_size = 1
    valid_kwargs = Videollama3FastImageProcessorKwargs
    model_input_names = [
        "pixel_values",
        "image_grid_thw",
        "image_merge_sizes",
        "pixel_values_videos",
        "video_grid_thw",
        "video_merge_sizes",
    ]

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        videos: VideoInput,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        **kwargs: Unpack[DefaultFastImageProcessorKwargs],
    ) -> BatchFeature:
        # Prepare input images
        batch_feature = BatchFeature()
        if images is not None:
            if kwargs["temporal_patch_size"] != 1:
                raise ValueError("`temporal_patch_size` must be 1 for VideoLLaMA3")
            images = self._prepare_image_like_inputs(
                images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
            )
            batch_feature = self._preprocess(images, **kwargs)
            batch_feature["image_merge_sizes"] = torch.tensor(
                [kwargs["merge_size"]] * batch_feature.image_grid_thw.size(0),
                dtype=batch_feature.image_grid_thw.dtype,
                device=batch_feature.image_grid_thw.device,
            )
        if videos is not None:
            logger.warning(
                "`Videollama3ImageProcessorFast` works only with image inputs and doesn't process videos anymore. "
                "This is a deprecated behavior and will be removed in v5.0. "
                "Your videos should be forwarded to `Videollama3VideoProcessor`. "
            )
            # Can't change _prepare_images_structure to work with videos because it also needs to work with images.
            videos = make_batched_videos(videos)
            videos = [
                torch.stack(self._prepare_image_like_inputs(video, do_convert_rgb, input_data_format, device))
                for video in videos
            ]
            video_outputs = self._preprocess(videos, **kwargs)
            batch_feature.update(
                {"pixel_values_videos": video_outputs.pixel_values, "video_grid_thw": video_outputs.image_grid_thw}
            )
            batch_feature["video_merge_sizes"] = torch.tensor(
                [kwargs["merge_size"]] * video_outputs.image_grid_thw.size(0),
                dtype=video_outputs.image_grid_thw.dtype,
                device=video_outputs.image_grid_thw.device,
            )
        return batch_feature


class Videollama3VideoProcessorInitKwargs(Qwen2VLVideoProcessorInitKwargs):
    use_token_compression: Optional[bool]


class Videollama3VideoProcessor(Qwen2VLVideoProcessor):
    use_token_compression = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    temporal_patch_size = 1
    max_frames = 180
    return_metadata = True
    valid_kwargs = Videollama3VideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw", "video_merge_sizes", "video_compression_mask"]

    def _get_compression_mask(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        video_merge_sizes: torch.LongTensor,
        threshold: Optional[float] = 0.1,
        min_tokens: Optional[int] = 1,
    ) -> torch.BoolTensor:
        """
        Get the compression mask for video tokens based on pixel differences.
        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            video_merge_sizes (`torch.Tensor` of shape `(num_videos,)`):
                The spatial downsampling ratio of each video feature.
            threshold (`float`, *optional*, defaults to 0.1):
                The threshold to determine whether a token should be kept based on pixel differences.
            min_tokens (`int`, *optional*, defaults to 1):
                The minimum number of tokens to keep for each frame.
        """
        videos = pixel_values_videos.split(video_grid_thw.prod(dim=1).tolist(), dim=0)
        compression_masks = []

        for images, grid_size, merge_size in zip(videos, video_grid_thw, video_merge_sizes):
            t, h, w = grid_size
            if t == 1:
                num_tokens = images.size(0) // (merge_size**2)
                compression_masks.append(torch.ones((num_tokens,), dtype=torch.bool, device=images.device))
            else:
                # NOTE: video token compressor
                images = images.view(t, (h // merge_size) * (w // merge_size), -1)

                pixel_diff = images[1:] - images[:-1]
                pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)
                mask = pixel_diff > threshold
                padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
                mask[padding_ids, :min_tokens] = 1
                compression_masks.append(mask.flatten())

        return torch.cat(compression_masks)

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        use_token_compression: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        device: Optional["torch.Tensor"] = None,
        **kwargs,
    ):
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            height, width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels // shape[0],
                )
                stacked_videos = self.resize(
                    image=stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            # Check that videos have `num_frames` divisible by `temporal_patch_size`
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, self.temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        video_merge_sizes = torch.tensor([merge_size] * video_grid_thw.size(0)).to(video_grid_thw)

        if use_token_compression:
            video_compression_mask = self._get_compression_mask(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                video_merge_sizes=video_merge_sizes,
            )
        else:
            num_video_tokens = video_grid_thw.prod(-1).sum() // (merge_size**2)
            video_compression_mask = torch.ones(
                (num_video_tokens,), dtype=torch.bool, device=pixel_values_videos.device
            )

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
                "video_merge_sizes": video_merge_sizes,
                "video_compression_mask": video_compression_mask,
            },
            tensor_type=return_tensors,
        )


__all__ = [
    "Videollama3VisionConfig",
    "Videollama3Config",
    "Videollama3VisionModel",
    "Videollama3Model",
    "Videollama3ForConditionalGeneration",
    "Videollama3Processor",
    "Videollama3ImageProcessorFast",
    "Videollama3VideoProcessor",
]
