# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
from collections.abc import Callable, Sequence
import enum
import itertools
import math
from typing import Literal, Optional, Union, cast

import PIL
import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...activations import ACT2FN
from ...cache_utils import Cache, HybridCache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import is_valid_image, make_flat_list_of_images
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    _validate_images_text_input_order,
)
from ...tokenization_utils_base import (
    AddedToken,
    TextInput,
)
from ...utils import is_torchdynamo_compiling, logging
from ..gemma.modeling_gemma import (
    GemmaAttention,
    GemmaForCausalLM,
    GemmaMLP,
    GemmaModel,
    GemmaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)
from ..siglip.configuration_siglip import SiglipVisionConfig


_CHECKPOINT_FOR_DOC = "google/gemma-3-4b"

logger = logging.get_logger(__name__)

IMAGE_TOKEN = "<image>"
EXTRA_TOKENS = [f"<loc{i:0>4}>" for i in range(1024)] + [f"<seg{i:0>3}>" for i in range(128)]

# Gemma 3 supports the following image input paradigms for any given prompt:
#
#   * No image      --> None
#   * Single-image  --> PIL.Image.Image
#   * Multi-image   --> Sequence[PIL.Image.Image]
#   * Batch         --> Sequence[Sequence[PIL.Image.Image]]
BatchedImageInput = Sequence[PIL.Image.Image]
BatchedMultiImageInput = Sequence[BatchedImageInput]
Gemma3ProcessorImageInput = Union[PIL.Image.Image, BatchedImageInput, BatchedMultiImageInput]

PanAndScannedImage = tuple[PIL.Image.Image, Sequence[PIL.Image.Image]]
BatchedPanAndScannedImage = Sequence[Sequence[PanAndScannedImage]]
MutablePanAndScannedImage = tuple[PIL.Image.Image, list[PIL.Image.Image]]
MutableBatchedPanAndScannedImage = list[list[MutablePanAndScannedImage]]

ATTENTION_TYPE_GLOBAL = "global_sliding"
ATTENTION_TYPE_LOCAL = "local_sliding"
AttentionType = Literal["global_sliding", "local_sliding"]
AttentionPattern = Sequence[AttentionType]
DEFAULT_ATTENION_PATTERN = cast(AttentionPattern, (
    ATTENTION_TYPE_LOCAL,
    ATTENTION_TYPE_LOCAL,
    ATTENTION_TYPE_LOCAL,
    ATTENTION_TYPE_LOCAL,
    ATTENTION_TYPE_LOCAL,
    ATTENTION_TYPE_GLOBAL,
))

TextInputTypes = Union[TextInput, Sequence[TextInput]]


class Gemma3TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma3Model`]. It is used to instantiate a Gemma3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma3-7B.
    e.g. [google/gemma-3-4b](https://huggingface.co/google/gemma-3-4b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemma3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Gemma3Model`]
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder. Will default to
            `"gelu_pytorch_tanh"` if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"`
            activation function.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_global_base_freq (float, *optional*, defaults to `rope_theta`):
            The base period of the RoPE embeddings for global attention.
        rope_local_base_freq (float, *optional*, defaults to `rope_theta`):
            The base period of the RoPE embeddings for local attention.
        attention_pattern (Sequence[AttentionTypes], defaults to (5 * local, global)):
            The attention pattern to apply
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256): scaling factor used on the attention scores
        sliding_window (`int`, *optional*, defaults to 4096): in Gemma3, every other layer uses sliding window
            attention. This is the size of the sliding window.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0): scaling factor when applying tanh soft-capping
            on the logits.z
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0): scaling factor when applying tanh soft-capping
            on the attention scorexs.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        cache_implementation (`str`, *optional*, defaults to `"hybrid"`): the cache type to be used with `generate`.

    ```python
    >>> from transformers import Gemma3Model, Gemma3Config
    >>> # Initializing a Gemma3 gemma3-7b style configuration
    >>> configuration = Gemma3Config()
    >>> # Initializing a model from the gemma3-7b style configuration
    >>> model = Gemma3Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "gemma3_text_model"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        # Config parameters found in all implementations, name differences noted
        vocab_size: int = 256000,                      # num_embed in FLAX
        hidden_size: int = 2304,                       # embed_dim in FLAX
        intermediate_size: int = 9216,                 # hidden_dim in FLAX
        num_hidden_layers: int = 26,                   # num_layers in FLAX
        num_attention_heads: int = 8,                  # num_heads in FLAX
        num_key_value_heads: int = 4,                  # num_kv_heads in FLAX
        head_dim: int = 256,
        sliding_window: int = 4096,                    # sliding_window_size in FLAX
        final_logit_softcapping: float = 30.0,
        query_pre_attn_scalar: int = 256,
        attention_pattern: AttentionPattern = DEFAULT_ATTENION_PATTERN,
        rope_theta: float = 10_000.0,                    # Consolidated in rope_wave_length Mapping in PyTorch
        rope_global_base_freq: float = 1_000_000.0,
        rope_local_base_freq: float = 10_000.0,
        rms_norm_eps: float = 1e-6,
        hidden_activation: str = "gelu_pytorch_tanh",
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        cache_implementation: str = "hybrid",

        # Config parameters still to be adjudicated
        use_pre_ffw_norm: bool = False,         # use_post_attn_norm in FLAX
        use_post_ffw_norm: bool = False,
        query_pre_attn_norm: Optional[enum.Enum] = None,
        compression_type: Optional[enum.Enum] = None,       # uant in Torch, v3_compression_type in FLAX
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_global_base_freq = rope_global_base_freq
        self.rope_local_base_freq = rope_local_base_freq
        self.attention_pattern = attention_pattern
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.cache_implementation = cache_implementation


class Gemma3VisionConfig(SiglipVisionConfig):

    def __init__(
        self,
        # SigLIP Vision Config Params
        hidden_size: int = 1152,                # width in FLAX
        intermediate_size: int = 4304,          # mlp_dim in FLAX
        num_hidden_layers: int = 27,            # depth in FLAX
        num_attention_heads: int = 16,          # num_heads in FLAX
        num_channels: int = 3,                  # image_channels in FLAX
        image_size: int = 896,                  # Split into image_height and image_width in FLAX
        attention_dropout: float = 0.0,         # dropout in FLAX
        patch_size: int = 14,
        # Config parameters in Transformers but not FLAX
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 0.000001,
        # Config parameters in FLAX but not Transformers
        position_embedding: str = "learn",
        representation_size: Union[int, bool] = False,
        pool_type: Optional[str] = "none",
        head_zeroinit: bool = True,
        scan: bool = False,
        remat_policy: str = "nothing_savable",
        output_length: int = 256,
        num_mm_tokens_per_image_prepool: int = 4096,
        num_mm_tokens_per_image: int = 256,
        apply_stop_gradient: bool = True,
        **kwargs
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            **kwargs,
        )

        self.position_embedding = position_embedding
        self.representation_size = representation_size
        self.pool_type = pool_type
        self.head_zeroinit = head_zeroinit
        self.scan = scan
        self.remat_policy = remat_policy
        self.output_length = output_length
        self.num_mm_tokens_per_image_prepool = num_mm_tokens_per_image_prepool
        self.num_mm_tokens_per_image = num_mm_tokens_per_image
        self.apply_stop_gradient = apply_stop_gradient


class Gemma3Config(PretrainedConfig):
    model_type = "gemma3"
    sub_configs = {
        "text_config": Gemma3TextConfig,
        "vision_config": Gemma3VisionConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        **kwargs
    ):
        if text_config is None:
            self.text_config = Gemma3TextConfig()
            logger.info("text_config is None, using default Gemma3TextConfig vision config.")
        elif isinstance(text_config, dict):
            self.text_config = Gemma3TextConfig(**text_config)
        elif isinstance(text_config, Gemma3TextConfig):
            self.text_config = text_config
        else:
            raise ValueError("text_config much be None or compatible with initializing a Gemma3TextConfig.")

        """
            Gemma 3 FLAX                SigLIP HF
            compression_type    ==>
            width               ==>     hidden_size
            mlp_dim             ==>     intermediate_size
            num_heads           ==>     num_attention_heads
            depth               ==>     num_hidden_layers
            patch_size          ==>     patch_size
            posemb              ==>
            rep_size            ==>
            dropout             ==>     attention_dropout
            pool_type           ==>
            head_zeroinit       ==>
            scan                ==>
            remat_policy        ==>
            dtype_mm            ==>
            output_length       ==>
        """
        if vision_config is None:
            self.vision_config = Gemma3VisionConfig()
            logger.info("vision_config is None, using default SigLIP vision config.")
        elif isinstance(vision_config, dict):
            self.vision_config = Gemma3VisionConfig(**vision_config)
        elif isinstance(vision_config, Gemma3VisionConfig):
            self.vision_config = vision_config

        super().__init__(**kwargs)


class Gemma3TextKwargs(TextKwargs):
    pass


class Gemma3ImagesKwargs(ImagesKwargs):
    do_pan_and_scan: bool
    pan_and_scan_min_crop_size: int
    pan_and_scan_max_num_crops: int
    pan_and_scan_min_ratio_to_activate: float
    do_convert_rgb: Optional[bool]
    do_resize: bool
    size: dict[str, int]
    resample: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
    do_rescale: bool = True,
    rescale_factor: Union[int, float] = 1 / 255,
    do_normalize: bool = True,
    image_mean: Optional[Union[float, list[float]]] = None,
    image_std: Optional[Union[float, list[float]]] = None,
    do_convert_rgb: bool = None,


class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Gemma3TextKwargs
    images_kwargs: Gemma3ImagesKwargs
    _defaults = {
        "text_kwargs": {"padding": False},
        "images_kwargs": {
            "data_format": "channels_first",
            "do_pan_and_scan": False,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
        },
    }


def pan_and_scan(
    image: PIL.Image.Image,
    pan_and_scan_min_crop_size: int,
    pan_and_scan_max_num_crops: int,
    pan_and_scan_min_ratio_to_activate: float,
    **unused_kwargs,
) -> Sequence[PIL.Image.Image]:
    w, h = image.size

    # Square or landscape image.
    if w >= h:
        # Only apply PaS if the image is sufficiently exaggerated
        if w / h < pan_and_scan_min_ratio_to_activate:
            return []

        # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
        num_crops_w = int(math.floor(w / h + 0.5))  # Half round up rounding.
        num_crops_w = min(int(math.floor(w / pan_and_scan_min_crop_size)), num_crops_w)

        # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
        num_crops_w = max(2, num_crops_w)
        num_crops_w = min(pan_and_scan_max_num_crops, num_crops_w)
        num_crops_h = 1

    # Portrait image.
    else:
        # Only apply PaS if the image is sufficiently exaggerated
        if h / w < pan_and_scan_min_ratio_to_activate:
            return []

        # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
        num_crops_h = int(math.floor(h / w + 0.5))
        num_crops_h = min(int(math.floor(h / pan_and_scan_min_crop_size)), num_crops_h)

        # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
        num_crops_h = max(2, num_crops_h)
        num_crops_h = min(pan_and_scan_max_num_crops, num_crops_h)
        num_crops_w = 1

    crop_size_w = int(math.ceil(w / num_crops_w))
    crop_size_h = int(math.ceil(h / num_crops_h))

    # Don't apply PaS if crop size is too small.
    if min(crop_size_w, crop_size_h) < pan_and_scan_min_crop_size:
        return []

    crop_positions_w = [crop_size_w * i  for i in range(num_crops_w)]
    crop_positions_h = [crop_size_h * i  for i in range(num_crops_h)]

    # Generate crops.
    return [
        image.crop((pos_w, pos_h, pos_w + crop_size_w, pos_h + crop_size_h))
        for pos_h, pos_w in itertools.product(crop_positions_h, crop_positions_w)
    ]


class Gemma3Processor(ProcessorMixin):

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        try:
            self.image_seq_length = getattr(image_processor, "image_seq_length")
        except AttributeError as e:
            raise ValueError("`image_processor` is missing the required `image_seq_length` attribute.") from e

        try:
            self.image_token_id = getattr(tokenizer, "image_token_id")
        except AttributeError:
            logger.warning("Image token not provided by `tokenizer`. Adding special `<image>` token.")

            image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [image_token]}
            tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        self.image_token = tokenizer.decode(self.image_token_id)

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    def __call__(
        self,
        images: Optional[Gemma3ProcessorImageInput] = None,
        text: Optional[TextInputTypes] = None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        # Check if images and text inputs are reversed for backward compatibility
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            images = self._process_images(images=images, **output_kwargs["images_kwargs"])

        text = self._process_text(text=text, **output_kwargs["text_kwargs"])

    def _process_images(
        self,
        images: Gemma3ProcessorImageInput,
        **kwargs: Unpack[Gemma3ImagesKwargs]
    ) -> BatchedPanAndScannedImage:
        # Normalize image structures
        if isinstance(images, PIL.Image.Image):
            images_lists: MutableBatchedPanAndScannedImage = [[(images, [])]]
        elif isinstance(images[0], PIL.Image.Image):
            images = cast(BatchedImageInput, images)
            images_lists: MutableBatchedPanAndScannedImage = [[(i, [])] for i in images]
        else:
            images = cast(BatchedMultiImageInput, images)
            images_lists: MutableBatchedPanAndScannedImage = [[(i, []) for i in il] for il in images]

        # if not all(len(images_lists[0]) == len(l) for l in images_lists):
        #     raise ValueError("All elements in a batch must have the same number of images.")

        if kwargs["do_pan_and_scan"]:
            if not isinstance(images_lists[0][0], PIL.Image.Image):
                raise ValueError("Pan and scan is only supported for `Pillow.Image.Image` inputs")

            for images_list in images_lists:
                for image, crops in images_list:
                    crops.extend(pan_and_scan(image=image, **kwargs))

        return images_lists


    def _process_text(
        self,
        text: Optional[TextInputTypes],
        images: Optional[BatchedPanAndScannedImage],
        **kwargs: Unpack[Gemma3TextKwargs]
    ) -> BatchFeature:
        return_value = BatchFeature()

        if images is not None:
            if text is None:
                pass
            else:
                pass

        inputs = self.tokenizer(text=text, **kwargs)
        return_value.update(inputs)
        return return_value


class Gemma3RMSNorm(GemmaRMSNorm):
    pass


class Gemma3MLP(GemmaMLP):
    def __init__(self, config):
        super().__init__()
        self.act_fn = ACT2FN[config.hidden_activation]


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Gemma3Attention(GemmaAttention):
    def __init__(self, config: Gemma3Config, layer_idx: int):

        super().__init__(config.text_config, layer_idx)

        self.attention_dropout = self.config.attention_dropout
        self.attention_type: AttentionType = config.text_config.attention_pattern[
            layer_idx % len(config.text_config.attention_pattern)
        ]
        self.is_causal = True
        self.scaling = config.text_config.query_pre_attn_scalar**-0.5
        self.sliding_window = config.text_config.sliding_window if not bool(layer_idx % 2) else None

        self.qk_norm = Gemma3RMSNorm(
            config.text_config.head_dim,
            config.text_config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.qk_norm(query_states)
        key_states = self.qk_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Here we need to slice as we use a static cache by default, but FA2 does not support it
            if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
                seq_len = attention_mask.shape[-1]
                key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = cast(
                    Callable[..., tuple[torch.Tensor, torch.Tensor]],
                    ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation],
                )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.text_config.hidden_size
        self.is_sliding = not bool(layer_idx % 2)
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.text_config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.text_config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.text_config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.text_config.rms_norm_eps)
        self.sliding_window = config.text_config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # In prefill, we may be larger than sliding window
            effective_seq_len = max(cache_position.shape[0], self.sliding_window)
            # For FA2, the mask is 2D and is of shape [bs, processed_tokens] (not [bs, max_cache_len]),
            # thus we must slice from the right (at most `effective_seq_len` elements)
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask[:, -effective_seq_len:]
            # Otherwise, the mask is 4D of shape [bs, 1, query_len, max_cache_len] thus we must slice
            # from the left, with an offset if we are beyond the sliding window
            else:
                min_dtype = torch.finfo(hidden_states.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool),
                    diagonal=-self.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                # In case we are beyond the sliding window, we need to correctly offset the mask slicing
                # `last_cache_position` is equivalent to `cache_position[-1]` but without breaking dynamo
                offset = last_cache_position - effective_seq_len
                # Should only be used when beyond the sliding window (i.e. offset > 0)
                offset = max(0, offset)
                attention_mask = attention_mask[:, :, :, offset : offset + effective_seq_len]

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Gemma3Model(GemmaModel):
    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: Optional[int] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = HybridCache(
                self.config,
                max_batch_size=batch_size,
                max_cache_len=seq_len,
                dtype=inputs_embeds.dtype,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # This is needed to correctly slice the mask without data-dependent slicing later on if using dynamo tracing
        # (retrieving the same value from `cache_position` later on would crash dynamo)
        if last_cache_position is None:
            last_cache_position = 0
            if attention_mask is not None:
                # In case a 4d mask is passed directly without using `generate`, we have to rely on cache_position
                # It will break dynamo tracing but there are no way around it (and it should never happen in practice)
                last_cache_position = (
                    attention_mask.shape[-1] if attention_mask.dim() == 2 else cache_position[-1].item()
                )
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # normalized
        # Gemma3 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.text_config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.text_config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    last_cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    last_cache_position=last_cache_position,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    @torch.no_grad()
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridCache,
        output_attentions: bool,
    ):
        # Flash Attention currently doesn't support static cache but Gemma3 work only with static cache.
        # So we will pass in attention mask as is in any case, not only when ther's padding. Then we'll use its shape
        # to cut out keys/values trailing 0 used in static cache. This workaround should be compile compatible
        # as it doesn't cause dynamic control issues.
        if self.config._attn_implementation == "flash_attention_2":
            return attention_mask

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        return causal_mask


class Gemma3ForCausalLM(GemmaForCausalLM):
    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.model = Gemma3Model(config)
        self.post_init()
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM

        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **loss_kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.text_config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

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
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten: has a special cache type, `HybridCache`

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        if past_key_values is not None:
            if (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s
                # `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride
                # during the decoding. Here, simply using `.contiguous()` is not sufficient as in the
                # batch size = 1 case, `position_ids` is already contiguous but with varying stride
                # which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        # This is needed to correctly slice the mask without data-dependent slicing later on if using dynamo tracing
        # (retrieving the same value from `cache_position` later on would crash dynamo)
        model_inputs["last_cache_position"] = attention_mask.shape[-1] if attention_mask is not None else 0

        if (
            isinstance(past_key_values, HybridCache)
            and attention_mask.ndim == 2
            and not self.config._attn_implementation == "flash_attention_2"
        ):
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


__all__ = [
    "Gemma3Config",
    "Gemma3ForCausalLM",
    "Gemma3Model",
    "Gemma3PreTrainedModel",  # noqa: F822
]
