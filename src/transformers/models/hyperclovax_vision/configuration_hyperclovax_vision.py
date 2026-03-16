# Copyright 2025 NAVER Corp. and the HuggingFace Inc. team. All rights reserved.
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
"""HyperClovaX model configuration"""

from ...configuration_utils import PretrainedConfig
from ...models.auto import CONFIG_MAPPING, AutoConfig
from ...models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class HyperClovaXConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HyperClovaXTextModel`]
    and [`HyperClovaXForCausalLM`]. It is used to instantiate a HyperClovaX language model
    according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    HyperClovaX-SEED-32B text backbone.
    e.g. [naver-hyperclovax/HyperCLOVAX-SEED-Think-32B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the
    model outputs. Read the documentation from [`PreTrainedConfig`] for more information.

    Args:
            vocab_size (`int`, *optional*, defaults to 32000):
                Vocabulary size of the model. Defines the number of different tokens that can be represented by
                `input_ids`.
            hidden_size (`int`, *optional*, defaults to 4096):
                Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 11008):
                Dimension of the MLP feed-forward representations.
            num_hidden_layers (`int`, *optional*, defaults to 32):
                Number of hidden layers in the Transformer decoder.
            num_attention_heads (`int`, *optional*, defaults to 32):
                Number of attention heads for each attention layer in the Transformer decoder.
            num_key_value_heads (`int | None`, *optional*):
                Number of key-value heads used for Grouped Query Attention (GQA). If `None`, defaults to
                `num_attention_heads` (standard multi-head attention).
            hidden_act (`str`, *optional*, defaults to `"silu"`):
                The non-linear activation function applied in the MLP layers.
            max_position_embeddings (`int`, *optional*, defaults to 2048):
                The maximum sequence length that this model can be used with.
            initializer_range (`float`, *optional*, defaults to 0.02):
                Standard deviation of the truncated normal distribution used to initialise all weight matrices.
            rms_norm_eps (`float`, *optional*, defaults to 1e-06):
                Epsilon value added to the denominator of RMSNorm layers for numerical stability.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether the model should cache past key-value states to speed up decoding. Disable during training.
            pad_token_id (`int | None`, *optional*):
                Token ID used for padding sequences to equal length in a batch.
            bos_token_id (`int`, *optional*, defaults to 1):
                Token ID representing the beginning of a sequence.
            eos_token_id (`int`, *optional*, defaults to 2):
                Token ID representing the end of a sequence.
            pretraining_tp (`int`, *optional*, defaults to 1):
                Tensor parallelism degree used during pretraining. Values greater than 1 activate Megatron-style
                tensor parallel linear layers for reproducibility.
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                Whether to tie the input token embedding weights to the output projection (lm_head) weights.
            rope_theta (`float`, *optional*, defaults to 10000.0):
                Base period of the Rotary Position Embedding (RoPE).
            rope_scaling (`dict | None`, *optional*):
                Dictionary containing RoPE scaling configuration. Supports keys `"rope_type"` (e.g. `"linear"`,
                `"dynamic"`, `"yarn"`) and scaling-specific hyperparameters. If `None`, no scaling is applied.
            attention_bias (`bool`, *optional*, defaults to `False`):
                Whether to include a learnable bias term in the query, key, value, and output projection layers.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                Dropout probability applied to attention weights.
            mlp_bias (`bool`, *optional*, defaults to `False`):
                Whether to include a learnable bias term in the MLP up-projection, gate-projection, and
                down-projection layers.
            head_dim (`int | None`, *optional*):
                Dimension of each attention head. Defaults to `hidden_size // num_attention_heads`.
            embedding_multiplier (`float`, *optional*, defaults to 1.0):
                Scalar multiplier applied to the token embeddings. Used for Maximal Update Parametrisation (MuP)
                to keep activation scale stable across model widths.
            logits_scaling (`float`, *optional*, defaults to 1.0):
                Scalar multiplier applied to the final logits before softmax. Used for MuP.
            attention_multiplier (`float`, *optional*, defaults to 1.0):
                Scalar multiplier applied to the attention scores (QK dot-products). Used for MuP.
            residual_multiplier (`float`, *optional*, defaults to 1.0):
                Scalar multiplier applied to each residual branch output before addition. Used for MuP.
            use_post_norm (`bool`, *optional*, defaults to `False`):
                Whether to apply an additional layer normalisation after each sub-layer output (dual-norm /
                post-norm). When `False`, only standard pre-norm is used.

    Example:

    ```python
    >>> from transformers import HyperClovaXConfig, HyperClovaXTextModel

    >>> # Initializing a HyperClovaX configuration
    >>> configuration = HyperClovaXConfig()

    >>> # Initializing a model from the configuration
    >>> model = HyperClovaXTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "hyperclovax"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        # Rotary positional embedding
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attention_multiplier: float = 1.0,
        # MuP parameters
        embedding_multiplier: float = 1.0,
        logits_scaling: float = 1.0,
        residual_multiplier: float = 1.0,
        # Post-norm (dual-norm)
        use_post_norm: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads

        # Rotary positional embedding params
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # If num_key_value_heads is not provided, default to num_attention_heads (standard MHA)
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # If head_dim is not provided, default to hidden_size // num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_multiplier = attention_multiplier
        # MuP
        self.embedding_multiplier = embedding_multiplier
        self.logits_scaling = logits_scaling
        self.residual_multiplier = residual_multiplier
        # Post-norm flag
        self.use_post_norm = use_post_norm

        super().__init__(**kwargs)


class HCXVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`HCXVisionForConditionalGeneration`]. It is used to instantiate a HyperClovaX Vision
    model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of
    HyperCLOVAX-SEED-Think-32B.
    e.g. [naver-hyperclovax/HyperCLOVAX-SEED-Think-32B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B)

    Combines a [`HyperClovaXConfig`] text backbone with a [`Qwen2_5_VLVisionConfig`] vision
    encoder and a configurable multimodal projector.

    The `sub_configs` mechanism automatically deserialises nested `text_config` and
    `vision_config` dicts when loading from a JSON config file, so a minimal hub
    `config.json` (with only `model_type`, `text_config`, and `vision_config` keys)
    is sufficient.

    Hub `config.json` compatibility notes:

    - `model_type` must be `"hyperclovax_vision"` (was `"vlm"` in the old trust_remote_code version).
    - `architectures` must reference the new class names (e.g. `["HCXVisionForConditionalGeneration"]`).
    - `text_config.model_type` should remain `"hyperclovax"` — unchanged.
    - `vision_config.model_type` can be either `"qwen2_5_vl"` (old value in the original hub
      config) or `"qwen2_5_vl_visual"` — both are accepted.
    - Deprecated path arguments (`text_model_name_or_path`, etc.) are silently ignored with a
      `FutureWarning`.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the
    model outputs. Read the documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`dict` or [`HyperClovaXConfig`], *optional*):
            Configuration for the text (LLM) component. Defaults to a standard [`HyperClovaXConfig`].
        vision_config (`dict` or [`Qwen2_5_VLVisionConfig`], *optional*):
            Configuration for the vision encoder component (Qwen2.5-VL ViT).
        use_nth_layer (`int`, *optional*, defaults to -2):
            Index of the vision encoder layer whose features are used as input to the projector.
            Negative indices count from the end (e.g., -2 = second-to-last layer).
        img_start_id (`int`, *optional*, defaults to 128060):
            Token ID used as a placeholder for image content in the input token sequence.
            Each image position in the text should contain this token ID. Also accessible as
            `image_token_id` for compatibility with generation helpers.
        video_start_id (`int`, *optional*, defaults to 128061):
            Token ID used as a placeholder for video content in the input token sequence.
            Also accessible as `video_token_id`.
        freeze_encoder (`bool`, *optional*, defaults to `False`):
            If `True`, the vision encoder weights are frozen during training.
        freeze_decoder (`bool`, *optional*, defaults to `False`):
            If `True`, the language model weights are frozen during training.
        freeze_mm_projector (`bool`, *optional*, defaults to `False`):
            If `True`, the multimodal projector weights are frozen during training.
        anyres (`bool`, *optional*, defaults to `False`):
            If `True`, enables any-resolution image processing where images are divided into
            variable-size grids based on their aspect ratio to minimize spatial information loss.
        unpad (`bool`, *optional*, defaults to `False`):
            If `True`, removes padding visual tokens from any-resolution processed features
            before passing to the language model, reducing unnecessary computation.
        max_num_grids (`int`, *optional*, defaults to -1):
            Maximum number of grid cells allowed when `anyres=True`. -1 means no limit.
        num_queries_vis_abstractor (`int`, *optional*, defaults to -1):
            Number of visual query tokens per grid for the CAbstractor projector. -1 means
            not used (CAbstractor not active).
        video_num_queries_fast (`int`, *optional*):
            Number of visual query tokens for fast (sub-sampled) video frames.
        video_num_queries_slow (`int`, *optional*):
            Number of visual query tokens for slow (full-resolution) video frames in SlowFast mode.
        video_first_last_frames_slows (`bool`, *optional*):
            If `True`, applies slow (full-resolution) processing only to the first and last frames.
        video_max_num_frames (`int`, *optional*):
            Maximum number of video frames to process. Frames beyond this limit are sub-sampled.
        ignore_index (`int`, *optional*, defaults to -100):
            Label index to ignore in the cross-entropy loss computation. Used to mask visual token
            positions in the labels tensor.
        proj_pos_emb (`bool`, *optional*, defaults to `True`):
            If `True`, learnable positional embeddings are added within the multimodal projector.
        proj_prenorm (`bool`, *optional*, defaults to `False`):
            If `True`, applies layer normalization before the multimodal projector.
        use_1x1_grid (`bool`, *optional*, defaults to `False`):
            If `True`, includes a 1×1 grid (single patch) as a possible resolution in anyres mode.
        possible_resolutions (`list`, *optional*, defaults to `[]`):
            Pre-computed list of `[height, width]` resolution pairs for anyres processing.

    Example:

    ```python
    >>> from transformers import HyperClovaXVisionConfig, HCXVisionForConditionalGeneration

    >>> # Initializing a HyperClovaX Vision configuration with defaults
    >>> configuration = HyperClovaXVisionConfig()

    >>> # Initializing a model from the configuration
    >>> model = HCXVisionForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "hyperclovax_vision"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"text_config": HyperClovaXConfig, "vision_config": Qwen2_5_VLVisionConfig}

    def __init__(
        self,
        text_config: AutoConfig | dict | None = None,
        vision_config: AutoConfig | dict | None = None,
        use_nth_layer: int = -2,
        img_start_id: int = 128060,
        video_start_id: int = 128061,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        freeze_mm_projector: bool = False,
        anyres: bool = False,
        unpad: bool = False,
        max_num_grids: int = -1,
        num_queries_vis_abstractor: int = -1,
        video_num_queries_fast: int | None = None,
        video_num_queries_slow: int | None = None,
        video_first_last_frames_slows: bool | None = None,
        video_max_num_frames: int | None = None,
        ignore_index: int = -100,
        proj_pos_emb: bool = True,
        proj_prenorm: bool = False,
        use_1x1_grid: bool = False,
        possible_resolutions: list | None = None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = HyperClovaXConfig()
        self.text_config = text_config

        if isinstance(vision_config, dict):
            if vision_config["architectures"][0] == "Qwen2_5_VisionTransformerPretrainedModel":
                vision_config = Qwen2_5_VLVisionConfig(**vision_config)
            else:
                vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = Qwen2_5_VLVisionConfig()
        self.vision_config = vision_config

        self.use_nth_layer = use_nth_layer
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_mm_projector = freeze_mm_projector
        self.anyres = anyres
        self.unpad = unpad
        self.max_num_grids = max_num_grids
        self.num_queries_vis_abstractor = num_queries_vis_abstractor

        # Video
        self.video_num_queries_fast = video_num_queries_fast
        self.video_num_queries_slow = video_num_queries_slow
        self.video_first_last_frames_slows = video_first_last_frames_slows
        self.video_max_num_frames = video_max_num_frames

        # Placeholder token IDs
        self.img_start_id = img_start_id
        self.image_token_id = img_start_id
        self.video_start_id = video_start_id
        self.video_token_id = video_start_id

        self.ignore_index = ignore_index
        self.proj_pos_emb = proj_pos_emb
        self.proj_prenorm = proj_prenorm
        self.use_1x1_grid = use_1x1_grid
        self.possible_resolutions = possible_resolutions if possible_resolutions is not None else []

        # Expose initializer_range at top level for _init_weights
        self.initializer_range = self.text_config.initializer_range

        if kwargs.get("model_type") == "vlm":
            kwargs["model_type"] = "hyperclovax_vision"

        super().__init__(**kwargs)


__all__ = ["HyperClovaXConfig", "HCXVisionConfig"]
