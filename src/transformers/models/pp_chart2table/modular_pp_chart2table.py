from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2.functional import InterpolationMode

from transformers.cache_utils import Cache
from transformers.configuration_utils import PreTrainedConfig, layer_type_validation
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation import GenerationMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.image_transforms import flip_channel_order, resize, to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_rope_utils import RopeParameters
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2Model, Qwen2PreTrainedModel
from transformers.processing_utils import ProcessorMixin, TensorType
from transformers.utils import (
    can_return_tuple,
    filter_out_non_signature_kwargs,
)


class PPChart2TableVisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PPChart2TableVisionModel`]. It is used to instantiate a
    PP-Chart2Table vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the PP-Chart2Table
    architecture developed by the PaddlePaddle team for chart-to-table parsing tasks.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        im_patch_token (`int`, *optional*, defaults to 151859):
            The token ID used to represent individual image patches in the multimodal input sequence.
        im_start_token (`int`, *optional*, defaults to 151857):
            The token ID representing the start of an image token sequence in the multimodal input.
        depth (`int`, *optional*, defaults to 12):
            Number of hidden layers in the vision Transformer encoder.
        embed_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the patch embedding layer in the vision encoder.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the hidden layers in the vision Transformer encoder.
        img_size (`int`, *optional*, defaults to 1024):
            The size (resolution) of input chart images (assumed to be square).
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of the dimensionality of the feed-forward layer to the hidden size in the vision Transformer blocks.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each self-attention layer in the vision Transformer encoder.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each image patch extracted from the input chart image.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias terms to the query, key, and value projection layers in the self-attention mechanism.
        use_rel_pos (`bool`, *optional*, defaults to `True`):
            Whether to use relative positional embeddings in the self-attention layers of the vision encoder.
        global_attn_indexes (`list`, *optional*, defaults to `[2, 5, 8, 11]`):
            List of layer indexes where global attention (instead of windowed attention) is applied in the vision encoder.
        window_size (`int`, *optional*, defaults to 14):
            The size of the attention window for windowed self-attention in the vision Transformer layers.
        out_chans (`int`, *optional*, defaults to 256):
            Number of output channels from the convolutional stem layer before patch embedding.

    Example:

    ```python
    >>> from transformers import PPChart2TableVisionConfig, PPChart2TableVisionModel

    >>> # Initializing a PPChart2TableVisionConfig with default PP-Chart2Table style configuration
    >>> configuration = PPChart2TableVisionConfig()

    >>> # Initializing a PPChart2TableVisionModel (with random weights) from the PP-Chart2Table style configuration
    >>> model = PPChart2TableVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    """

    model_type = "pp_chart2table_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        im_patch_token: int = 151859,
        im_start_token: int = 151857,
        depth: int = 12,
        embed_dim: int = 768,
        hidden_size: int = 1024,
        img_size: int = 1024,
        mlp_ratio: float = 4.0,
        num_heads: int = 12,
        patch_size: int = 16,
        qkv_bias: bool = True,
        use_rel_pos: bool = True,
        global_attn_indexes: Optional[list] = None,
        window_size: int = 14,
        out_chans: int = 256,
        **kwargs,
    ):
        self.im_patch_token = im_patch_token
        self.im_start_token = im_start_token

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.global_attn_indexes = global_attn_indexes if global_attn_indexes is not None else [2, 5, 8, 11]
        self.window_size = window_size
        self.out_chans = out_chans

        super().__init__(**kwargs)


class PPChart2TableTextConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PPChart2TableTextModel`]. It is used to instantiate a
    PP-Chart2Table text decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text encoder/decoder of the
    PPChart2TableText-7B-beta [Qwen/PPChart2TableText-7B-beta](https://huggingface.co/Qwen/PPChart2TableText-7B-beta)
    architecture, optimized for chart-to-table text generation tasks.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities in self-attention layers.
        bos_token_id (`int`, *optional*, defaults to 151643):
            The token ID representing the beginning of a sequence (BOS) for text generation.
        eos_token_id (`int`, *optional*, defaults to 151643):
            The token ID representing the end of a sequence (EOS) for text generation.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the feed-forward and attention layers of the decoder.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the hidden representations in the Transformer decoder layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        intermediate_size (`int`, *optional*, defaults to 2816):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer decoder blocks.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with for text input/output.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each self-attention layer in the Transformer decoder.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key/value heads for implementing Grouped Query Attention (GQA). If equal to `num_attention_heads`,
            Multi Head Attention (MHA) is used; if 1, Multi Query Attention (MQA) is used. For more details, see
            [this paper](https://huggingface.co/papers/2305.13245).
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon value used by the RMS normalization layers to avoid division by zero.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE (Rotary Position Embedding) embeddings, controlling the frequency of positional encoding.
        rope_parameters (`RopeParameters` or `dict`, *optional*):
            Configuration parameters for RoPE embeddings, including scaling parameters for longer sequence lengths beyond
            `max_position_embeddings`.
        sliding_window (`int`, *optional*, defaults to 32768):
            Window size for Sliding Window Attention (SWA) in the decoder layers (only active if `use_sliding_window=True`).
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied (shared weights).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the last key/value attention states to speed up sequential decoding (only relevant for autoregressive
            generation).
        vocab_size (`int`, *optional*, defaults to 151860):
            Vocabulary size of the PPChart2TableText model. Defines the number of distinct tokens that can be represented
            by `input_ids`.
        layer_types (`list[str]`, *optional*):
            Attention pattern for each decoder layer (e.g., `"full_attention"` or `"sliding_attention"`). If not specified,
            automatically determined by `sliding_window`.

    Example:

    ```python
    >>> from transformers import PPChart2TableTextConfig, PPChart2TableTextModel

    >>> # Initializing a PPChart2TableText style configuration
    >>> configuration = PPChart2TableTextConfig()

    >>> # Initializing a model from the PPChart2TableText-7B style configuration
    >>> model = PPChart2TableTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    """

    model_type = "pp_chart2table_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `PPChart2TableText`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_config_key = "text_config"

    def __init__(
        self,
        attention_dropout: float = 0.0,
        bos_token_id: int = 151643,
        eos_token_id: int = 151643,
        hidden_act: str = "silu",
        hidden_size: int = 1024,
        initializer_range: float = 0.02,
        intermediate_size: int = 2816,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 24,
        num_key_value_heads: int = 16,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        sliding_window: int = 32768,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        vocab_size: int = 151860,
        layer_types: Optional[list[str]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.attention_dropout = attention_dropout

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if self.sliding_window is not None else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters

        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class PPChart2TableConfig(PreTrainedConfig):
    r"""
    This is the main configuration class to store the configuration of a [`PPChart2TableModel`] or [`PPChart2TableForConditionalGeneration`].
    It is used to instantiate a PP-Chart2Table multimodal model according to the specified arguments, defining the vision and text
    sub-model architectures. This configuration class inherits from [`PreTrainedConfig`] and combines the configurations of:
    - [`PPChart2TableVisionConfig`] (for the chart vision encoder)
    - [`PPChart2TableTextConfig`] (for the table text decoder)
    PP-Chart2Table [PaddlePaddle/PP-Chart2Table_safetensors](https://huggingface.co/PaddlePaddle/PP-Chart2Table_safetensors).

    Instantiating a `PPChart2TableConfig` with the defaults will yield a similar configuration to the base PP-Chart2Table model
    developed by the PaddlePaddle team for chart-to-table parsing tasks.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`PPChart2TableVisionConfig`]. If `None`, the default
            `PPChart2TableVisionConfig` configuration will be used.
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`PPChart2TableTextConfig`]. If `None`, the default
            `PPChart2TableTextConfig` configuration will be used.
        im_start_token (`int`, *optional*, defaults to 151857):
            The token ID representing the start of an image token sequence in the multimodal input (shared across vision/text sub-configs).
        im_patch_token (`int`, *optional*, defaults to 151859):
            The token ID used to represent individual image patches in the multimodal input sequence (shared across vision/text sub-configs).

    Example:

    ```python
    >>> from transformers import PPChart2TableConfig, PPChart2TableModel

    >>> # Initializing a PPChart2Table configuration with default vision and text sub-configs
    >>> configuration = PPChart2TableConfig()

    >>> # Initializing a PPChart2Table configuration with custom vision and text sub-configs
    >>> vision_config = {"img_size": 512, "patch_size": 8}
    >>> text_config = {"hidden_size": 2048, "num_hidden_layers": 16}
    >>> configuration = PPChart2TableConfig(vision_config=vision_config, text_config=text_config)

    >>> # Initializing a model from the PPChart2Table configuration
    >>> model = PPChart2TableModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> # Accessing the vision sub-config
    >>> vision_config = configuration.vision_config
    >>> # Accessing the text sub-config
    >>> text_config = configuration.text_config
    """

    model_type = "pp_chart2table"
    sub_configs = {"vision_config": PPChart2TableVisionConfig, "text_config": PPChart2TableTextConfig}

    def __init__(
        self,
        vision_config: dict | None = None,
        text_config: dict | None = None,
        im_start_token: int = 151857,
        im_patch_token: int = 151859,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = {}
        self.vision_config = PPChart2TableVisionConfig(**vision_config)

        if text_config is None:
            text_config = {}
        self.text_config = PPChart2TableTextConfig(**text_config)

        self.model_type = "pp_chart2table"

        self.im_start_token = im_start_token
        self.im_patch_token = im_patch_token

        text_config_keys = [
            "attention_dropout",
            "bos_token_id",
            "eos_token_id",
            "hidden_act",
            "hidden_size",
            "initializer_range",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "sliding_window",
            "tie_word_embeddings",
            "dtype",
            "use_cache",
            "vocab_size",
        ]
        for key in text_config_keys:
            if hasattr(self.text_config, key):
                setattr(self, key, getattr(self.text_config, key))

        super().__init__(**kwargs)


class PPChart2TableImageProcessor(BaseImageProcessor):
    r"""
    Image processor for the PP-Chart2Table multimodal model, optimized for chart image preprocessing tasks.

    This processor handles the complete preprocessing pipeline for chart images, including resizing, rescaling,
    normalization, and channel dimension reordering, tailored to the input requirements of the PP-Chart2Table vision encoder.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input images to the specified `size`.
        size (`dict[str, int]`, *optional*, defaults to `{"height": 256, "width": 256}`):
            Dictionary containing the target height and width for resizing. Format: `{"height": int, "width": int}`.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use when resizing images (e.g., BICUBIC, BILINEAR).
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the pixel values from the range [0, 255] to [0, 1] using `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Factor to apply for rescaling pixel values (e.g., 1/255 scales 0-255 to 0-1).
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the input images using `image_mean` and `image_std`.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.406, 0.456, 0.485]`):
            Mean values for image normalization (per channel, RGB order).
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.225, 0.224, 0.229]`):
            Standard deviation values for image normalization (per channel, RGB order).
        patch_size (`int`, *optional*, defaults to 16):
            Size of image patches used by the PP-Chart2Table vision encoder (for alignment with model input).
        merge_size (`int`, *optional*, defaults to 4):
            Size factor for merging image patches (specific to PP-Chart2Table's vision processing pipeline).
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = [0.406, 0.456, 0.485],
        image_std: Optional[Union[float, list[float]]] = [0.225, 0.224, 0.229],
        patch_size: int = 16,
        merge_size: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 256, "width": 256}

        self.do_resize = do_resize
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.resample = resample
        self.patch_size = patch_size
        self.merge_size = merge_size

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        size: Optional[dict[str, int]] = None,
        do_resize: Optional[bool] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        size = self.size if size is None else size
        do_resize = self.do_resize if do_resize is None else do_resize
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std

        images = make_flat_list_of_images(images)

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            size=size,
            do_resize=do_resize,
            resample=resample,
        )

        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        # transformations
        resize_imgs = []
        if do_resize:
            for image in images:
                img = resize(
                    image,
                    size=(size["height"], size["width"]),
                    resample=resample,
                    input_data_format=input_data_format,
                )
                resize_imgs.append(img)
            images = resize_imgs

        if do_rescale:
            images = [self.rescale(image, rescale_factor, input_data_format=input_data_format) for image in images]

        if do_normalize:
            images = [
                self.normalize(image, image_mean, image_std, input_data_format=input_data_format) for image in images
            ]
        images = [flip_channel_order(image, input_data_format=input_data_format) for image in images]
        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        encoded_inputs = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)
        return encoded_inputs


class PPChart2TableImageProcessorFast(BaseImageProcessorFast):
    r"""
    Fast image processor for the PP-Chart2Table multimodal model, optimized for GPU-accelerated chart image preprocessing.

    This high-performance processor implements a streamlined preprocessing pipeline for chart images (resizing, rescaling,
    normalization, channel reordering) using PyTorch tensor operations, designed for efficient batch processing on GPUs.
    It inherits from [`BaseImageProcessorFast`] and is optimized for inference/training pipelines requiring low-latency
    image preprocessing.

    Class Attributes (Default Configuration):
        resample (`int`, defaults to 3):
            Integer identifier for the resampling filter (3 = BICUBIC, compatible with `InterpolationMode.BICUBIC`).
        image_mean (`list[float]`, defaults to `[0.40821073, 0.4578275, 0.48145466]`):
            Per-channel mean values for image normalization (RGB order).
        image_std (`list[float]`, defaults to `[0.27577711, 0.26130258, 0.26862954]`):
            Per-channel standard deviation values for image normalization (RGB order).
        size (`dict[str, int]`, defaults to `{"height": 1024, "width": 1024}`):
            Default target size for image resizing (1024x1024, optimized for PP-Chart2Table vision encoder).
        patch_size (`int`, defaults to 16):
            Size of image patches used by the PP-Chart2Table vision encoder (for alignment with model input).
        merge_size (`int`, defaults to 4):
            Size factor for merging image patches (specific to PP-Chart2Table's vision processing pipeline).
        do_resize (`bool`, defaults to `True`):
            Default flag to enable image resizing.
        do_rescale (`bool`, defaults to `True`):
            Default flag to enable pixel value rescaling (from [0,255] to [0,1]).
        do_normalize (`bool`, defaults to `True`):
            Default flag to enable image normalization.
    """

    resample = 3
    image_mean = [0.40821073, 0.4578275, 0.48145466]
    image_std = [0.27577711, 0.26130258, 0.26862954]
    size = {"height": 1024, "width": 1024}
    patch_size = 16
    merge_size = 4
    do_resize = True
    do_rescale = True
    do_normalize = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        size: Optional[list[dict[str, int]]],
        do_resize: bool,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        interpolation: Optional[InterpolationMode] = None,
        **kwargs,
    ) -> BatchFeature:
        data = {}
        resize_imgs = []
        if do_resize:
            for image in images:
                img = self.resize(image, size=size, interpolation=interpolation)
                resize_imgs.append(img)
            images = resize_imgs

        processed_images = []
        for image in images:
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            processed_images.append(image)
        images = processed_images

        images = [image[[2, 1, 0], :, :] for image in images]
        data.update({"pixel_values": torch.stack(images, dim=0)})
        encoded_inputs = BatchFeature(data, tensor_type=return_tensors)

        return encoded_inputs


class PPChart2TableProcessor(ProcessorMixin):
    r"""
    [`PPChart2TableProcessor`] offers all the functionalities of [`PPChart2TableImageProcessor`] and [`Qwen2Tokenizer`]. See the
    [`~PPChart2TableProcessor.__call__`] and [`~PPChart2TableProcessor.decode`] for more information.
    Args:
        image_processor ([`PPChart2TableImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2Tokenizer`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images,
        text=None,
        **kwargs,
    ) -> BatchFeature:
        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors="pt")
        else:
            image_inputs = {}
        img_cnt = len(image_inputs)
        b, c, h, w = image_inputs["pixel_values"].shape
        num_patches = h // self.image_processor.patch_size // self.image_processor.merge_size
        prompt = (
            "<|im_start|>system\n"
            "You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user\n"
            "<img>" + "<imgpad>" * (num_patches * num_patches) + "</img>\n"
            "Chart to table<|im_end|><|im_start|>assistant\n"
        )
        input_ids = torch.tensor(self.tokenizer([prompt]).input_ids)
        input_ids = input_ids.repeat(img_cnt, 1)
        input_ids = {"input_ids": input_ids}
        return BatchFeature(data={**input_ids, **image_inputs})

    def postprocess(self, model_pred, **kwargs):
        return self.tokenizer.batch_decode(
            model_pred[0],
            skip_special_tokens=kwargs.get("skip_special_tokens", True),
            clean_up_tokenization_spaces=False,
        )


def window_partition(hidden_states: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    r"""
    Partition 2D feature maps into non-overlapping windows, with padding to ensure dimensions are divisible by window size.

    Args:
        hidden_states (`torch.Tensor`):
            Input feature map with shape [B, H, W, C], where:
            - B: batch size
            - H: height of feature map
            - W: width of feature map
            - C: channel dimension
        window_size (`int`):
            Size of each non-overlapping window (square window).

    Returns:
        tuple[torch.Tensor, tuple[int, int]]:
            - windows: Partitioned windows with shape [num_windows * B, window_size, window_size, C],
              where num_windows = (Hp // window_size) * (Wp // window_size)
            - (Hp, Wp): Padded height and width of the feature map (after padding)
    """
    B, H, W, C = hidden_states.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    hidden_states = hidden_states.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = hidden_states.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> torch.Tensor:
    r"""
    Reverse operation of window_partition: merge windows back to original 2D feature map shape, removing padding.

    Args:
        windows (`torch.Tensor`):
            Partitioned windows with shape [num_windows * B, window_size, window_size, C]
        window_size (`int`):
            Size of each non-overlapping window (must match window_partition's window_size)
        pad_hw (`tuple[int, int]`):
            Padded height and width (Hp, Wp) returned by window_partition
        hw (`tuple[int, int]`):
            Original height and width (H, W) of feature map before padding

    Returns:
        `torch.Tensor`:
            Reconstructed feature map with shape [B, H, W, C] (original dimensions before padding)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    hidden_states = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        hidden_states = hidden_states[:, :H, :W, :]
    return hidden_states


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    r"""
    Get relative positional embeddings for query and key sequences, with interpolation for mismatched sizes.

    Args:
        q_size (`int`):
            Spatial size (height/width) of query feature map
        k_size (`int`):
            Spatial size (height/width) of key feature map
        rel_pos (`torch.Tensor`):
            Precomputed relative positional embeddings with shape [max_rel_dist_original, dim]

    Returns:
        `torch.Tensor`:
            Interpolated relative positional embeddings for the query-key pair, shape [q_size, k_size, dim]
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> torch.Tensor:
    r"""
    Add decomposed relative positional embeddings (height and width separately) to attention scores.

    Args:
        attn (`torch.Tensor`):
            Attention scores with shape [B, q_h*q_w, k_h*k_w]
        q (`torch.Tensor`):
            Query tensor with shape [B, q_h*q_w, dim]
        rel_pos_h (`torch.Tensor`):
            Precomputed relative positional embeddings for height dimension
        rel_pos_w (`torch.Tensor`):
            Precomputed relative positional embeddings for width dimension
        q_size (`tuple[int, int]`):
            Spatial size (q_h, q_w) of query feature map
        k_size (`tuple[int, int]`):
            Spatial size (k_h, k_w) of key feature map

    Returns:
        `torch.Tensor`:
            Attention scores with added relative positional embeddings, shape [B, q_h*q_w, k_h*k_w]
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.reshape(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).reshape(
        B, q_h * q_w, k_h * k_w
    )

    return attn


class PPChart2TableVisionPatchEmbed(nn.Module):
    r"""
    Image to Patch Embedding layer for PP-Chart2Table vision encoder.

    This module converts raw chart images (HWC format) into flattened patch embeddings via a 2D convolution,
    followed by dimension permutation to align with the vision transformer's input format.

    Args:
        kernel_size (`tuple[int, int]`, *optional*, defaults to `(16, 16)`):
            Size of the convolution kernel (patch size) for splitting images into patches.
        stride (`tuple[int, int]`, *optional*, defaults to `(16, 16)`):
            Stride of the convolution operation (matches patch size for non-overlapping patches).
        padding (`tuple[int, int]`, *optional*, defaults to `(0, 0)`):
            Padding applied to the input image before convolution (ensures patch alignment).
        in_chans (`int`, *optional*, defaults to 3):
            Number of input channels (3 for RGB chart images).
        embed_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the output patch embeddings (hidden size of the vision transformer).

    Shape:
        - Input: `(B, C, H, W)` (batch size, channels, height, width)
        - Output: `(B, H_out, W_out, C_out)` (batch size, patch height, patch width, embedding dim)
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        return hidden_states


class PPChart2TableVisionMLPBlock(nn.Module):
    r"""
    Multi-Layer Perceptron (MLP) block for PP-Chart2Table vision transformer layers.

    Implements a two-layer feed-forward network with activation function, used in the vision transformer's
    decoder layers to project features to a higher dimension and back.

    Args:
        embedding_dim (`int`):
            Dimensionality of the input/output embeddings (hidden size of the transformer layer).
        mlp_dim (`int`):
            Dimensionality of the intermediate (hidden) layer in the MLP (typically 4x embedding_dim).
        act (`Type[nn.Module]`, *optional*, defaults to `torch.nn.GELU`):
            Non-linear activation function to apply between the two linear layers.

    Shape:
        - Input: `(B, H, W, embedding_dim)` or `(B, N, embedding_dim)` (N = H*W)
        - Output: Same shape as input
    """

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: type[nn.Module] = torch.nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(hidden_states)))


class PPChart2TableVisionLayerNorm2d(nn.Module):
    r"""
    2D Layer Normalization for spatial feature maps (adapted for PP-Chart2Table vision encoder).

    Applies layer normalization over the channel dimension of 2D feature maps, with learnable scale/bias parameters
    broadcasted across spatial dimensions (height/width).

    Args:
        num_channels (`int`):
            Number of channels in the input feature map (embedding dimension).
        epsilon (`float`, *optional*, defaults to `1e-06`):
            Small value added to variance to avoid division by zero.

    Shape:
        - Input: `(B, C, H, W)` (batch size, channels, height, width)
        - Output: Same shape as input
    """

    def __init__(self, num_channels: int, epsilon: float = 1e-06) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        u = hidden_states.mean(dim=1, keepdim=True)
        s = (hidden_states - u).pow(2).mean(dim=1, keepdim=True)
        hidden_states = (hidden_states - u) / torch.sqrt(s + self.epsilon)
        hidden_states = self.weight[:, None, None] * hidden_states + self.bias[:, None, None]
        return hidden_states


class PPChart2TableVisionAttention(nn.Module):
    r"""
    Multi-Head Self-Attention (MHSA) layer for PP-Chart2Table vision encoder, with optional relative positional encoding.

    Implements standard multi-head attention with query/key/value projection, scaled dot-product attention,
    and optional decomposed relative positional embeddings (height/width separate) for spatial awareness.

    Args:
        dim (`int`):
            Dimensionality of the input embeddings (hidden size of the transformer layer).
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads (must divide `dim` evenly).
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias terms to the query/key/value projection layers.
        use_rel_pos (`bool`, *optional*, defaults to `False`):
            Whether to use relative positional encoding for spatial attention.
        rel_pos_zero_init (`bool`, *optional*, defaults to `True`):
            Whether to initialize relative positional embeddings to zero (stable training).
        input_size (`Tuple[int, int]`, *optional*):
            Spatial size (H, W) of the input feature map (required if `use_rel_pos=True`).

    Shape:
        - Input: `(B, H, W, dim)` (batch size, height, width, embedding dim)
        - Output: Same shape as input
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(dim=0)
        attn = (q * self.scale) @ k.transpose(1, 2)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = F.softmax(attn, dim=-1)
        hidden_states = (attn @ v).reshape(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class PPChart2TableVisionDecoderLayer(nn.Module):
    r"""
    Single decoder layer of the PP-Chart2Table vision transformer, with optional windowed attention.

    Implements the standard transformer decoder layer structure:
    Layer Norm → Multi-Head Attention (with residual) → Layer Norm → MLP (with residual)
    Supports windowed attention (SW-MHA) for large feature maps to reduce computation.

    Args:
        dim (`int`):
            Dimensionality of the input embeddings (hidden size of the transformer layer).
        num_heads (`int`):
            Number of attention heads (passed to MHSA layer).
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of MLP hidden dimension to embedding dimension (mlp_dim = dim * mlp_ratio).
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in Q/K/V projection (passed to MHSA layer).
        norm_layer (`Type[nn.Module]`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer to use (LayerNorm for flattened patches, LayerNorm2d for 2D feature maps).
        act_layer (`Type[nn.Module]`, *optional*, defaults to `nn.GELU`):
            Activation function for MLP block.
        use_rel_pos (`bool`, *optional*, defaults to `False`):
            Whether to use relative positional encoding (passed to MHSA layer).
        rel_pos_zero_init (`bool`, *optional*, defaults to `True`):
            Whether to zero-initialize relative positional embeddings (passed to MHSA layer).
        window_size (`int`, *optional*, defaults to 0):
            Size of attention windows (0 = full attention, >0 = windowed attention).
        input_size (`Tuple[int, int]`, *optional*):
            Spatial size of input feature map (passed to MHSA layer for relative positional encoding).

    Shape:
        - Input: `(B, H, W, dim)` (batch size, height, width, embedding dim)
        - Output: Same shape as input
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PPChart2TableVisionAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = PPChart2TableVisionMLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shortcut = hidden_states
        hidden_states = self.norm1(hidden_states)
        if self.window_size > 0:
            H, W = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_hw = window_partition(hidden_states, self.window_size)
        hidden_states = self.attn(hidden_states)

        if self.window_size > 0:
            hidden_states = window_unpartition(hidden_states, self.window_size, pad_hw, (H, W))
        hidden_states = shortcut + hidden_states
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class PPChart2TableVisionPreTrainedModel(PreTrainedModel):
    r"""
    Base class for all PP-Chart2Table vision models, inheriting from Hugging Face `PreTrainedModel`.

    This class sets up core configurations and compatibility flags for the vision encoder, including:
    - Support for gradient checkpointing, attention backends (FlashAttention/SDPA), and model compilation
    - Definition of non-splittable modules (for tensor parallelism)
    - Output recording for hidden states/attentions (for debugging/analysis)

    Class Attributes:
        config (`PPChart2TableVisionConfig`):
            Typed config class for PP-Chart2Table vision encoder (enforces type checking).
        base_model_prefix (`str`, defaults to `"model"`):
            Prefix for base model parameters (used in weight loading/saving).
        supports_gradient_checkpointing (`bool`, defaults to `True`):
            Whether the model supports gradient checkpointing to save memory.
        _no_split_modules (`list[str]`):
            Modules that should not be split across devices (tensor parallelism compatibility).
        _skip_keys_device_placement (`list[str]`):
            Keys to skip when placing tensors on devices (e.g., past key values for generation).
        _supports_flash_attn / _supports_sdpa / _supports_flex_attn (`bool`):
            Compatibility with optimized attention implementations (FlashAttention, SDPA, FlexAttention).
        _can_compile_fullgraph (`bool`, defaults to `True`):
            Whether the model supports TorchScript/TorchCompile full graph compilation.
        _supports_attention_backend (`bool`, defaults to `True`):
            Whether the model supports switching attention backends (e.g., PyTorch vs FlashAttention).
        _can_record_outputs (`dict`):
            Mapping of output types to modules for recording intermediate outputs (hidden_states/attentions).
    """

    config: PPChart2TableVisionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PPChart2TableVisionDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": PPChart2TableVisionDecoderLayer,
        "attentions": PPChart2TableVisionAttention,
    }


class PPChart2TableVisionModel(PPChart2TableVisionPreTrainedModel):

    main_input_name = "pixel_values"
    input_modalities = "image"

    def __init__(
        self,
        config: PPChart2TableVisionConfig,
        in_chans: int = 3,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        rel_pos_zero_init: bool = True,
    ) -> None:
        super().__init__(config)
        self.img_size = config.img_size

        self.patch_embed = PPChart2TableVisionPatchEmbed(
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
            in_chans=in_chans,
            embed_dim=config.embed_dim,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, config.img_size // config.patch_size, config.img_size // config.patch_size, config.embed_dim
            )
        )

        self.blocks = nn.ModuleList()
        for i in range(config.depth):
            block = PPChart2TableVisionDecoderLayer(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=config.use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
                input_size=(config.img_size // config.patch_size, config.img_size // config.patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                config.embed_dim,
                config.out_chans,
                kernel_size=1,
                bias=False,
            ),
            PPChart2TableVisionLayerNorm2d(config.out_chans),
            nn.Conv2d(
                config.out_chans,
                config.out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            PPChart2TableVisionLayerNorm2d(config.out_chans),
        )

        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, config.hidden_size, kernel_size=3, stride=2, padding=1, bias=False)

        self.post_init()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.pos_embed
        for blk in self.blocks:
            hidden_states = blk(hidden_states)
        hidden_states = self.neck(hidden_states.permute(0, 3, 1, 2))
        hidden_states = self.net_2(hidden_states)
        hidden_states = self.net_3(hidden_states)
        return hidden_states

class PPChart2TableTextAttention(Qwen2Attention):
    pass


class PPChart2TableTextDecoderLayer(Qwen2DecoderLayer):
    pass


class PPChart2TableTextPreTrainedModel(Qwen2PreTrainedModel):
    pass

class PPChart2TableTextModel(Qwen2Model):
    pass


@dataclass
class PPChart2TableModelOutputWithPast(ModelOutput):
    r"""
    Output class for PPChart2Table multimodal model's forward pass, extending Hugging Face `ModelOutput`.

    This dataclass encapsulates the core outputs of the PP-Chart2Table base model, including hidden states,
    attention weights, and cached key/value pairs for efficient generation.

    Attributes:
        past_key_values (`Optional[Cache]`, defaults to `None`):
            Cached attention key/value pairs from the text decoder (for fast autoregressive generation).
        last_hidden_state (`Optional[torch.FloatTensor]`, defaults to `None`):
            Final hidden states from the text decoder (shape: `[B, seq_len, hidden_size]`), after multimodal fusion.
        hidden_states (`Optional[tuple[torch.FloatTensor]]`, defaults to `None`):
            Tuple of hidden states from each layer of the text decoder (for debugging/analysis).
        attentions (`Optional[tuple[torch.FloatTensor]]`, defaults to `None`):
            Tuple of attention weights from each layer of the text decoder (for debugging/analysis).
    """

    past_key_values: Optional[Cache] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
class PPChart2TableCausalLMOutputWithPast(ModelOutput):
    r"""
    Output class for PP-Chart2Table conditional generation model's forward pass.

    Extends `PPChart2TableModelOutputWithPast` with language modeling logits (for token prediction),
    tailored for autoregressive table generation tasks.

    Attributes:
        logits (`Optional[torch.FloatTensor]`, defaults to `None`):
            Language modeling logits (shape: `[B, seq_len, vocab_size]`), output from the LM head.
        past_key_values (`Optional[Cache]`, defaults to `None`):
            Cached attention key/value pairs (inherited from base model output).
        last_hidden_state (`Optional[torch.FloatTensor]`, defaults to `None`):
            Final hidden states from the text decoder (inherited from base model output).
        hidden_states (`Optional[tuple[torch.FloatTensor]]`, defaults to `None`):
            Tuple of decoder layer hidden states (inherited from base model output).
        attentions (`Optional[tuple[torch.FloatTensor]]`, defaults to `None`):
            Tuple of decoder layer attention weights (inherited from base model output).
    """

    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class PPChart2TablePreTrainedModel(PreTrainedModel):
    r"""
    Base class for all PP-Chart2Table multimodal models, inheriting from Hugging Face `PreTrainedModel`.

    This class defines core configurations and compatibility flags for the multimodal model (vision + text),
    including support for gradient checkpointing, optimized attention backends, and model compilation.

    Class Attributes:
        config (`PPChart2TableConfig`):
            Typed config class for PP-Chart2Table (combines vision + text sub-configs).
        base_model_prefix (`str`, defaults to `"model"`):
            Prefix for base model parameters (used in weight loading/saving).
        supports_gradient_checkpointing (`bool`, defaults to `True`):
            Whether the model supports gradient checkpointing to save memory during training.
        _no_split_modules (`list[str]`):
            Modules that should not be split across devices (tensor parallelism compatibility).
        _skip_keys_device_placement (`list[str]`):
            Keys to skip when placing tensors on devices (e.g., past key values for generation).
        _supports_flash_attn / _supports_sdpa / _supports_flex_attn (`bool`):
            Compatibility with optimized attention implementations (FlashAttention, SDPA, FlexAttention).
        _can_compile_fullgraph (`bool`, defaults to `True`):
            Whether the model supports TorchScript/TorchCompile full graph compilation.
        _supports_attention_backend (`bool`, defaults to `True`):
            Whether the model supports switching attention backends (e.g., PyTorch vs FlashAttention).
        _can_record_outputs (`dict`):
            Mapping of output types to modules for recording intermediate outputs (hidden_states/attentions).
    """

    config: PPChart2TableConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PPChart2TableTextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": PPChart2TableTextDecoderLayer,
        "attentions": PPChart2TableTextAttention,
    }


class PPChart2TableModel(PPChart2TablePreTrainedModel):
    r"""
    Core PP-Chart2Table multimodal model (vision encoder + text decoder) for chart-to-table parsing.

    This model integrates a vision encoder (for chart image feature extraction) and a text decoder (for table generation),
    with a multimodal projection layer to align vision features with text embedding space. The core logic is:
    1. Extract chart features via vision encoder
    2. Project vision features to text embedding dimension
    3. Inject vision features into text decoder inputs (replace image placeholder tokens)
    4. Forward pass through text decoder to generate table text

    Args:
        config (`PPChart2TableConfig`):
            Combined configuration class (includes vision_config and text_config sub-configs).

    Inputs (forward method):
        input_ids (`torch.LongTensor`, optional):
            Tokenized input text (including image placeholder tokens) with shape `[B, seq_len]`.
        attention_mask (`torch.Tensor`, optional):
            Attention mask to avoid padding tokens (shape: `[B, seq_len]`).
        position_ids (`torch.Tensor`, optional):
            Positional indices for input tokens (shape: `[B, seq_len]`).
        past_key_values (`list[torch.Tensor]`, optional):
            Cached key/value pairs for fast autoregressive generation.
        inputs_embeds (`torch.Tensor`, optional):
            Precomputed input embeddings (shape: `[B, seq_len, hidden_size]`; overrides `input_ids`).
        use_cache (`bool`, optional):
            Whether to cache key/value pairs for generation.
        pixel_values (`torch.Tensor`, optional):
            Preprocessed chart images (shape: `[B, 3, H, W]`; required for multimodal input).
        cache_position (`torch.LongTensor`, optional):
            Position indices for cached key/value pairs (for generation).
        **kwargs:
            Additional arguments passed to the text decoder.

    Outputs:
        `PPChart2TableModelOutputWithPast`:
            Contains the text decoder's final hidden states, cached key/values, and optional intermediate outputs.
    """

    config_class = PPChart2TableConfig

    def __init__(self, config: PPChart2TableConfig):
        super().__init__(config)
        self.vision_tower_high = PPChart2TableVisionModel._from_config(config.vision_config)
        self.language_model = PPChart2TableTextModel._from_config(config.text_config)
        self.mm_projector_vary = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """Get input embeddings from the text decoder (for weight tying/loading)."""
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        """Set input embeddings for the text decoder (for weight tying/loading)."""
        self.language_model.embed_tokens = value

    def get_image_features(
        self,
        images: Optional[torch.Tensor],
    ) -> list[torch.Tensor]:
        r"""
        Extract and project chart image features to text embedding space.

        Args:
            images (`torch.Tensor`):
                Preprocessed chart images (shape: `[B, 3, H, W]`).

        Returns:
            `list[torch.Tensor]`:
                List of projected image features (one per image), each with shape `[1, num_patches, text_hidden_size]`.
        """
        image_features = []
        for image in images:
            image = image.unsqueeze(0)
            with torch.no_grad():
                cnn_feature = self.vision_tower_high(image)
                cnn_feature = cnn_feature.flatten(2).transpose(2, 1)
            image_feature = self.mm_projector_vary(cnn_feature)
            image_features.append(image_feature)

        image_features = torch.stack(image_features, dim=0)

        return image_features

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
    ) -> torch.BoolTensor:
        r"""
        Generate mask to locate image placeholder tokens in input embeddings.

        This mask identifies the `<imgpad>` tokens in the input sequence, which will be replaced with
        projected image features for multimodal fusion.

        Args:
            input_ids (`torch.LongTensor`, optional):
                Tokenized input text (used if `inputs_embeds` is None).
            inputs_embeds (`torch.FloatTensor`, optional):
                Precomputed input embeddings (used if `input_ids` is None).
            image_features (`torch.FloatTensor`):
                Projected image features (used to validate token-feature count match).

        Returns:
            `torch.BoolTensor`:
                Boolean mask (shape: `[B, seq_len, text_hidden_size]`) where `True` indicates image placeholder tokens.

        Raises:
            ValueError: If the number of image tokens does not match the number of image features.
        """
        if input_ids is None:
            start_token_embed = self.get_input_embeddings()(
                torch.tensor(self.config.im_start_token, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = inputs_embeds == start_token_embed
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.im_patch_token

        n_image_tokens = special_image_mask.sum()

        n_image_features = image_features.numel() // image_features.shape[-1]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        return special_image_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        output = PPChart2TableModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return output


class PPChart2TableForConditionalGeneration(PPChart2TablePreTrainedModel, GenerationMixin):
    r"""
    PP-Chart2Table model for conditional generation (table text generation from chart images),
    extending the core model with a language modeling (LM) head and generation utilities.

    This class integrates Hugging Face `GenerationMixin` to support standard generation methods (greedy, beam search, etc.),
    and adds an LM head to predict token probabilities for autoregressive table generation.

    Key Features:
    - LM head for token prediction (weight tied to input embeddings)
    - Optimized generation input preparation (avoids reprocessing images in subsequent steps)
    - Inference-only mode (training not supported by default)

    Args:
        config (`PPChart2TableConfig`):
            Combined configuration class (vision + text sub-configs).

    Inputs (forward method):
        Inherits all inputs from `PPChart2TableModel`, plus:
        labels (`list[dict]`, optional):
            Training labels (not supported; raises ValueError if provided).
        logits_to_keep (`Union[int, torch.Tensor]`, defaults to 0):
            Slice index to keep only the last N logits (optimizes generation efficiency).

    Outputs:
        `PPChart2TableCausalLMOutputWithPast`:
            Contains LM logits, decoder hidden states, and cached key/value pairs.
    """

    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: PPChart2TableConfig):
        super().__init__(config)
        self.model = PPChart2TableModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
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
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None

        return model_inputs

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[list[dict]] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], PPChart2TableCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            raise ValueError(
                "The PPChart2TableForConditionalGeneration model only supports inference, and training is not allowed!\n"
                "If you need to train this model, please implement the corresponding loss calculation logic, or use the inference-only mode (do not pass the `labels` parameter)."
            )

        return PPChart2TableCausalLMOutputWithPast(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "PPChart2TableForConditionalGeneration",
    "PPChart2TableModel",
    "PPChart2TablePreTrainedModel",
    "PPChart2TableConfig",
    "PPChart2TableTextPreTrainedModel",
    "PPChart2TableTextModel",
    "PPChart2TableVisionPreTrainedModel",
    "PPChart2TableVisionModel",
    "PPChart2TableVisionConfig",
    "PPChart2TableTextConfig",
    "PPChart2TableImageProcessor",
    "PPChart2TableImageProcessorFast",
    "PPChart2TableProcessor",
]
