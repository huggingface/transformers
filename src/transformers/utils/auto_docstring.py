# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import UnionType
from typing import ClassVar, Union, get_args, get_origin

import regex as re
import typing_extensions

from .doc import (
    MODELS_TO_PIPELINE,
    PIPELINE_TASKS_TO_SAMPLE_DOCSTRINGS,
    PT_SAMPLE_DOCSTRINGS,
)
from .generic import ModelOutput


PATH_TO_TRANSFORMERS = Path("src").resolve() / "transformers"


AUTODOC_FILES = [
    "configuration_*.py",
    "modeling_*.py",
    "tokenization_*.py",
    "processing_*.py",
    "image_processing_pil_*.py",
    "image_processing_*.py",
    "feature_extractor_*.py",
]

PLACEHOLDER_TO_AUTO_MODULE = {
    "image_processor_class": ("image_processing_auto", "IMAGE_PROCESSOR_MAPPING_NAMES"),
    "tokenizer_class": ("tokenization_auto", "TOKENIZER_MAPPING_NAMES"),
    "video_processor_class": ("video_processing_auto", "VIDEO_PROCESSOR_MAPPING_NAMES"),
    "feature_extractor_class": ("feature_extraction_auto", "FEATURE_EXTRACTOR_MAPPING_NAMES"),
    "processor_class": ("processing_auto", "PROCESSOR_MAPPING_NAMES"),
    "config_class": ("configuration_auto", "CONFIG_MAPPING_NAMES"),
    "model_class": ("modeling_auto", "MODEL_MAPPING_NAMES"),
}

UNROLL_KWARGS_METHODS = {
    "preprocess",
    "__call__",
}

UNROLL_KWARGS_CLASSES = {
    "BaseImageProcessor",
    "ProcessorMixin",
}
BASIC_KWARGS_TYPES = ["TextKwargs", "ImagesKwargs", "VideosKwargs", "AudioKwargs"]

# Short indicator added to unrolled kwargs to distinguish them from regular args
KWARGS_INDICATOR = ", *kwargs*"

HARDCODED_CONFIG_FOR_MODELS = {
    "openai": "OpenAIGPTConfig",
    "x-clip": "XCLIPConfig",
    "kosmos2": "Kosmos2Config",
    "kosmos2-5": "Kosmos2_5Config",
    "donut": "DonutSwinConfig",
    "esmfold": "EsmConfig",
    "parakeet": "ParakeetCTCConfig",
    "privacy-filter": "OpenAIPrivacyFilterConfig",
    "lasr": "LasrCTCConfig",
    "wav2vec2-with-lm": "Wav2Vec2Config",
}

_re_checkpoint = re.compile(r"\[(.+?)\]\((https://huggingface\.co/.+?)\)")

# Pre-compiled patterns used repeatedly at runtime.  Compiling once here avoids
# repeated compilation overhead (and cache lookups) on every decorator call.
_re_example_or_return = re.compile(r"(?m)^([ \t]*)(?=Example|Return|```)")
_re_return = re.compile(r"(?m)^([ \t]*)(?=Return)")
_re_example = re.compile(r"(?m)^([ \t]*)(?=Example|```)")
_re_args_section = re.compile(r"(?:Args:)(\n.*)?(\n)?$", re.DOTALL)
_re_shape = re.compile(r"(of shape\s*(?:`.*?`|\(.*?\)))")
_re_default = re.compile(r"(defaults to \s*[^)]*)")
_re_param = re.compile(
    r"^\s{0,0}(\w+)\s*\(\s*([^, \)]*)(\s*.*?)\s*\)\s*:\s*((?:(?!\n^\s{0,0}\w+\s*\().)*)",
    re.DOTALL | re.MULTILINE,
)
_re_forward_ref = re.compile(r"ForwardRef\('([\w.]+)'\)")
_re_optional = re.compile(r"Optional\[(.*?)\]")
_re_placeholders = re.compile(r"{(.*?)}")
_re_model_task = None  # built lazily because PT_SAMPLE_DOCSTRINGS isn't available yet


class ImageProcessorArgs:
    images = {
        "description": """
    Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
    passing in images with pixel values between 0 and 1, set `do_rescale=False`.
    """,
        "shape": None,
    }

    videos = {
        "description": """
    Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
    passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
    """,
        "shape": None,
    }

    do_resize = {
        "description": """
    Whether to resize the image.
    """,
        "shape": None,
    }

    size = {
        "description": """
    Describes the maximum input dimensions to the model.
    """,
        "shape": None,
    }

    size_divisor = {
        "description": """
    The size by which to make sure both the height and width can be divided.
    """,
        "shape": None,
    }

    default_to_square = {
        "description": """
    Whether to default to a square image when resizing, if size is an int.
    """,
        "shape": None,
    }

    resample = {
        "description": """
    Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
    has an effect if `do_resize` is set to `True`.
    """,
        "shape": None,
    }

    do_center_crop = {
        "description": """
    Whether to center crop the image.
    """,
        "shape": None,
    }

    crop_size = {
        "description": """
    Size of the output image after applying `center_crop`.
    """,
        "shape": None,
    }

    do_pad = {
        "description": """
    Whether to pad the image. Padding is done either to the largest size in the batch
    or to a fixed square size per image. The exact padding strategy depends on the model.
    """,
        "shape": None,
    }

    pad_size = {
        "description": """
    The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
        provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
        height and width in the batch. Applied only when `do_pad=True.`
    """,
        "shape": None,
    }

    do_rescale = {
        "description": """
    Whether to rescale the image.
    """,
        "shape": None,
    }

    rescale_factor = {
        "description": """
    Rescale factor to rescale the image by if `do_rescale` is set to `True`.
    """,
        "shape": None,
    }

    do_normalize = {
        "description": """
    Whether to normalize the image.
    """,
        "shape": None,
    }

    image_mean = {
        "description": """
    Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
    """,
        "shape": None,
    }

    image_std = {
        "description": """
    Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
    `True`.
    """,
        "shape": None,
    }

    do_convert_rgb = {
        "description": """
    Whether to convert the image to RGB.
    """,
        "shape": None,
    }

    return_tensors = {
        "description": """
    Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.
    """,
        "shape": None,
    }

    data_format = {
        "description": """
    Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
    """,
        "shape": None,
    }

    input_data_format = {
        "description": """
    The channel dimension format for the input image. If unset, the channel dimension format is inferred
    from the input image. Can be one of:
    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
    - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
    """,
        "shape": None,
    }

    device = {
        "description": """
    The device to process the images on. If unset, the device is inferred from the input images.
    """,
        "shape": None,
    }

    disable_grouping = {
        "description": """
    Whether to disable grouping of images by size to process them individually and not in batches.
    If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
    empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157
    """,
        "shape": None,
    }

    image_seq_length = {
        "description": """
    The number of image tokens to be used for each image in the input.
    Added for backward compatibility but this should be set as a processor attribute in future models.
    """,
        "shape": None,
    }

    # Used for the **kwargs summary line when unrolling typed kwargs (key: "__kwargs__")
    __kwargs__ = {
        "description": """
    Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class
    for the complete list of supported arguments.
    """,
        "shape": None,
    }


class ProcessorArgs:
    # __init__ arguments
    image_processor = {
        "description": """
    The image processor is a required input.
    """,
        "type": "{image_processor_class}",
    }

    tokenizer = {
        "description": """
    The tokenizer is a required input.
    """,
        "type": "{tokenizer_class}",
    }

    video_processor = {
        "description": """
    The video processor is a required input.
    """,
        "type": "{video_processor_class}",
    }

    audio_processor = {
        "description": """
    The audio processor is a required input.
    """,
        "type": "{audio_processor_class}",
    }

    feature_extractor = {
        "description": """
    The feature extractor is a required input.
    """,
        "type": "{feature_extractor_class}",
    }

    chat_template = {
        "description": """
    A Jinja template to convert lists of messages in a chat into a tokenizable string.
    """,
        "type": "str",
    }

    # __call__ arguments
    text = {
        "description": """
    The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
    (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.
    """,
    }

    audio = {
        "description": """
    The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor.
    In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
    and T is the sample length of the audio.
    """,
    }

    audios = {
        "description": """
    The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor.
    In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
    and T is the sample length of the audio.
    """,
    }

    return_tensors = {
        "description": """
    If set, will return tensors of a particular framework. Acceptable values are:

    - `'pt'`: Return PyTorch `torch.Tensor` objects.
    - `'np'`: Return NumPy `np.ndarray` objects.
    """,
        "shape": None,
    }

    # Standard tokenizer arguments
    add_special_tokens = {
        "description": """
    Whether or not to add special tokens when encoding the sequences. This will use the underlying
    [`PretrainedTokenizerBase.build_inputs_with_special_tokens`] function, which defines which tokens are
    automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
    automatically.
    """,
        "type": "bool",
    }

    padding = {
        "description": """
    Activates and controls padding. Accepts the following values:

    - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
      sequence is provided).
    - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
      acceptable input length for the model if that argument is not provided.
    - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
      lengths).
    """,
        "type": "bool, str or [`~utils.PaddingStrategy`]",
    }

    truncation = {
        "description": """
    Activates and controls truncation. Accepts the following values:

    - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
      to the maximum acceptable input length for the model if that argument is not provided. This will
      truncate token by token, removing a token from the longest sequence in the pair if a pair of
      sequences (or a batch of pairs) is provided.
    - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
      maximum acceptable input length for the model if that argument is not provided. This will only
      truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
      maximum acceptable input length for the model if that argument is not provided. This will only
      truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
      greater than the model maximum admissible input size).
    """,
        "type": "bool, str or [`~tokenization_utils_base.TruncationStrategy`]",
    }

    max_length = {
        "description": """
    Controls the maximum length to use by one of the truncation/padding parameters.

    If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
    is required by one of the truncation/padding parameters. If the model has no specific maximum input
    length (like XLNet) truncation/padding to a maximum length will be deactivated.
    """,
        "type": "int",
    }

    stride = {
        "description": """
    If set to a number along with `max_length`, the overflowing tokens returned when
    `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
    returned to provide some overlap between truncated and overflowing sequences. The value of this
    argument defines the number of overlapping tokens.
    """,
        "type": "int",
    }

    pad_to_multiple_of = {
        "description": """
    If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
    This is especially useful to enable using Tensor Cores on NVIDIA hardware with compute capability
    `>= 7.5` (Volta).
    """,
        "type": "int",
    }

    return_token_type_ids = {
        "description": """
    Whether to return token type IDs. If left to the default, will return the token type IDs according to
    the specific tokenizer's default, defined by the `return_outputs` attribute.

    [What are token type IDs?](../glossary#token-type-ids)
    """,
        "type": "bool",
    }

    return_attention_mask = {
        "description": """
    Whether to return the attention mask. If left to the default, will return the attention mask according
    to the specific tokenizer's default, defined by the `return_outputs` attribute.

    [What are attention masks?](../glossary#attention-mask)
    """,
        "type": "bool",
    }

    return_overflowing_tokens = {
        "description": """
    Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
    of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
    of returning overflowing tokens.
    """,
        "type": "bool",
    }

    return_special_tokens_mask = {
        "description": """
    Whether or not to return special tokens mask information.
    """,
        "type": "bool",
    }

    return_offsets_mapping = {
        "description": """
    Whether or not to return `(char_start, char_end)` for each token.

    This is only available on fast tokenizers inheriting from [`PreTrainedTokenizerFast`], if using
    Python's tokenizer, this method will raise `NotImplementedError`.
    """,
        "type": "bool",
    }

    return_length = {
        "description": """
    Whether or not to return the lengths of the encoded inputs.
    """,
        "type": "bool",
    }

    verbose = {
        "description": """
    Whether or not to print more information and warnings.
    """,
        "type": "bool",
    }

    text_pair = {
        "description": """
    Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
    the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
    method).
    """,
        "type": "str, list[str] or list[int]",
    }

    text_target = {
        "description": """
    The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
    list of strings (pretokenized string). If you pass pretokenized input, set `is_split_into_words=True`
    to avoid ambiguity with batched inputs.
    """,
        "type": "str, list[str] or list[list[str]]",
    }

    text_pair_target = {
        "description": """
    The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
    list of strings (pretokenized string). If you pass pretokenized input, set `is_split_into_words=True`
    to avoid ambiguity with batched inputs.
    """,
        "type": "str, list[str] or list[list[str]]",
    }

    is_split_into_words = {
        "description": """
    Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
    tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
    which it will tokenize. This is useful for NER or token classification.
    """,
        "type": "bool",
    }

    boxes = {
        "description": """
    Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.
    """,
        "type": "list[list[int]] or list[list[list[int]]]",
    }

    word_labels = {
        "description": """
    Word-level integer labels (for token classification tasks such as FUNSD, CORD).
    """,
        "type": "list[int] or list[list[int]]",
    }

    # Used for the **kwargs summary line when unrolling typed kwargs (key: "__kwargs__")
    __kwargs__ = {
        "description": """
    Additional processing options for each modality (text, images, videos, audio). Model-specific parameters
    are listed above; see the TypedDict class for the complete list of supported arguments.
    """,
        "shape": None,
    }


class ConfigArgs:
    output_hidden_states = {
        "description": """
    Whether or not the model should return all hidden-states.
    """,
    }

    chunk_size_feed_forward = {
        "description": """
    The `dtype` of the weights. This attribute can be used to initialize the model to a non-default `dtype`
    (which is normally `float32`) and thus allow for optimal storage allocation. For example, if the saved
    model is `float16`, ideally we want to load it back using the minimal amount of memory needed to load
    `float16` weights.
    """,
    }

    dtype = {
        "description": """
    The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means that
    the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes `n` <
    sequence_length embeddings at a time. For more information on feed forward chunking, see [How does Feed
    Forward Chunking work?](../glossary.html#feed-forward-chunking).
    """,
    }

    id2label = {
        "description": """
    A map from index (for instance prediction index, or target index) to label.
    """,
    }

    label2id = {
        "description": """
    A map from label to index for the model.
    """,
    }

    problem_type = {
        "description": """
    Problem type for `XxxForSequenceClassification` models. Can be one of `"regression"`,
            `"single_label_classification"` or `"multi_label_classification"`.
    """,
    }

    tokenizer_class = {
        "description": """
    The class name of model's tokenizer.
    """,
    }

    vocab_size = {
        "description": """
    Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.
    """,
    }

    hidden_size = {
        "description": """
    Dimension of the hidden representations.
    """,
    }

    intermediate_size = {
        "description": """
    Dimension of the MLP representations.
    """,
    }

    head_dim = {
        "description": """
    The attention head dimension. If None, it will default to hidden_size // num_attention_heads
    """
    }

    num_hidden_layers = {
        "description": """
    Number of hidden layers in the Transformer decoder.
    """,
    }

    num_attention_heads = {
        "description": """
    Number of attention heads for each attention layer in the Transformer decoder.
    """,
    }

    num_key_value_heads = {
        "description": """
    This is the number of key_value heads that should be used to implement Grouped Query Attention. If
    `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
    `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
    converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
    by meanpooling all the original heads within that group. For more details, check out [this
    paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
    `num_attention_heads`.
    """,
    }
    hidden_act = {
        "description": """
    The non-linear activation function (function or string) in the decoder. For example, `"gelu"`,
    `"relu"`, `"silu"`, etc.
    """,
    }

    max_position_embeddings = {
        "description": """
    The maximum sequence length that this model might ever be used with.
    """,
    }

    initializer_range = {
        "description": """
    The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """,
    }

    rms_norm_eps = {
        "description": """
    The epsilon used by the rms normalization layers.
    """,
    }

    use_cache = {
        "description": """
    Whether or not the model should return the last key/values attentions (not used by all models). Only
    relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.
    """,
    }

    rope_parameters = {
        "description": """
    Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
    a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
    with longer `max_position_embeddings`.
    """,
    }

    attention_bias = {
        "description": """
    Whether to use a bias in the query, key, value and output projection layers during self-attention.
    """,
    }

    mlp_bias = {
        "description": """
    Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
    """,
    }

    attention_dropout = {
        "description": """
    The dropout ratio for the attention probabilities.
    """,
    }

    pretraining_tp = {
        "description": """
    Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
    document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
    understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
    results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
    """,
    }

    pad_token_id = {
        "description": """
    Token id used for padding in the vocabulary.
    """,
    }

    eos_token_id = {
        "description": """
    Token id used for end-of-stream in the vocabulary.
    """,
    }

    bos_token_id = {
        "description": """
    Token id used for beginning-of-stream in the vocabulary.
    """,
    }

    sep_token_id = {
        "description": """
    Token id used for separator in the vocabulary.
    """,
    }

    cls_token_id = {
        "description": """
    Token id used for CLS in the vocabulary.
    """,
    }

    tie_word_embeddings = {
        "description": """
    Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.
    """,
    }

    d_model = {
        "description": """
    Size of the encoder layers and the pooler layer.
    """,
    }

    d_kv = {
        "description": """
    Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will
    be defined as `num_heads * d_kv`.
    """,
    }

    num_decoder_layers = {
        "description": """
    Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
    """,
    }

    num_encoder_layers = {
        "description": """
    Number of hidden layers in the Transformer encoder. Will use the same value as `num_layers` if not set.
    """,
    }

    dropout_rate = {
        "description": """
    The ratio for all dropout layers.
    """,
    }

    classifier_dropout = {
        "description": """
    The dropout ratio for classifier.
    """,
    }

    layer_norm_eps = {
        "description": """
    The epsilon used by the layer normalization layers.
    """,
    }

    initializer_factor = {
        "description": """
    A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
    testing).
    """,
    }

    encoder_attention_heads = {
        "description": """
    Number of attention heads for each attention layer in the Transformer encoder.
    """,
    }

    decoder_attention_heads = {
        "description": """
    Number of attention heads for each attention layer in the Transformer decoder.
    """,
    }

    decoder_ffn_dim = {
        "description": """
    Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    """,
    }

    encoder_ffn_dim = {
        "description": """
    Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
    """,
    }

    activation_dropout = {
        "description": """
    The dropout ratio for activations inside the fully connected layer.
    """,
    }

    encoder_layerdrop = {
        "description": """
    The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556)
    for more details.
    """,
    }

    decoder_layerdrop = {
        "description": """
    The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556)
    for more details.
    """,
    }

    scale_embedding = {
        "description": """
    Whether to scale embeddings by dividing by sqrt(d_model).
    """,
    }

    forced_eos_token_id = {
        "description": """
    The id of the token to force as the last generated token when `max_length` is reached. Usually set to
    `eos_token_id`.
    """,
    }

    moe_intermediate_size = {
        "description": """
    Intermediate size of the routed expert MLPs.
    """,
    }

    num_experts = {
        "description": """
    Number of routed experts in MoE layers.

    """,
    }

    num_experts_per_tok = {
        "description": """
    Number of experts to route each token to. This is the top-k value for the token-choice routing.
    """,
    }

    num_shared_experts = {
        "description": """
    Number of shared experts that are always activated for all tokens.
    """,
    }

    layer_types = {
        "description": """
    A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically
    generated based on config values.
    """,
    }

    norm_topk_prob = {
        "description": """
    Whether to normalize the weights of the routed experts.

    """,
    }

    topk_group = {
        "description": """
    Number of selected groups for each token (for each token, ensuring the selected experts is only within `topk_group` groups).
    """,
    }

    qk_rope_head_dim = {
        "description": """
    Dimension of the query/key heads that use rotary position embeddings.
    """,
    }

    v_head_dim = {
        "description": """
    Dimension of the value heads.
    """,
    }

    qk_nope_head_dim = {
        "description": """
    Dimension of the query/key heads that don't use rotary position embeddings.
    """,
    }

    kv_lora_rank = {
        "description": """
    Rank of the LoRA matrices for key and value projections.
    """,
    }

    q_lora_rank = {
        "description": """
    Rank of the LoRA matrices for query projections.
    """,
    }

    routed_scaling_factor = {
        "description": """
    Scaling factor or routed experts.
    """,
    }

    n_routed_experts = {
        "description": """
    Number of routed experts.
    """,
    }

    n_shared_experts = {
        "description": """
    Number of shared experts.
    """,
    }

    vision_config = {
        "description": """
    The config object or dictionary of the vision backbone.
    """,
    }

    text_config = {
        "description": """
    The config object or dictionary of the text backbone.
    """,
    }

    projector_hidden_act = {
        "description": """
    The activation function used by the multimodal projector.
    """,
    }

    vision_feature_select_strategy = {
        "description": """
    The feature selection strategy used to select the vision feature from the vision backbone.
    """,
    }

    vision_feature_layer = {
        "description": """
    The index of the layer to select the vision feature. If multiple indices are provided,
    the vision feature of the corresponding indices will be concatenated to form the
    vision features.
    """,
    }

    multimodal_projector_bias = {
        "description": """
    Whether to use bias in the multimodal projector.
    """,
    }

    image_token_id = {
        "description": """
    The image token index used as a placeholder for input images.
    """,
    }

    video_token_id = {
        "description": """
    The video token index used as a placeholder for input videos.
    """,
    }

    audio_token_id = {
        "description": """
    The audio token index used as a placeholder for input audio.
    """,
    }

    image_seq_length = {
        "description": """
    Sequence length of one image embedding.
    """,
    }

    video_seq_length = {
        "description": """
    Sequence length of one video embedding.
    """,
    }

    add_cross_attention = {
        "description": """
    Whether cross-attention layers should be added to the model.
    """,
    }

    is_decoder = {
        "description": """
    Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    """,
    }

    sliding_window = {
        "description": """
    Sliding window attention window size. If `None`, no sliding window is applied.
    """,
    }

    use_sliding_window = {
        "description": """
    Whether to use sliding window attention.
    """,
    }

    shared_expert_intermediate_size = {
        "description": """
    Intermediate size of the shared expert MLPs.
    """,
    }

    decoder_sparse_step = {
        "description": """
    The frequency of adding a sparse MoE layer. The default is 1, which means all decoder layers are sparse MoE.
    """,
    }

    output_router_logits = {
        "description": """
    Whether or not the router logits should be returned by the model. Enabling this will also allow the model
    to output the auxiliary loss, including load balancing loss and router z-loss.
    """,
    }

    router_aux_loss_coef = {
        "description": """
    Auxiliary load balancing loss coefficient. Used to penalize uneven expert routing in MoE models.
    """,
    }

    out_indices = {
        "description": """
    Indices of the intermediate hidden states (feature maps) to return from the backbone. Each index
    corresponds to one stage of the model.
    """,
    }

    out_features = {
        "description": """
    Names of the intermediate hidden states (feature maps) to return from the backbone. One of `"stem"`,
    `"stage1"`, `"stage2"`, etc.
    """,
    }

    image_size = {
        "description": """
    The size (resolution) of each image.
    """,
    }

    patch_size = {
        "description": """
    The size (resolution) of each patch.
    """,
    }

    num_channels = {
        "description": """
    The number of input channels.
    """,
    }

    num_mel_bins = {
        "description": """
    Number of mel features used per input frame. Should correspond to the value used in the
    `AutoFeatureExtractor` class.
    """,
    }

    sampling_rate = {
        "description": """
    The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
    """,
    }

    hidden_dropout = {
        "description": """
    The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    """,
    }

    mlp_ratio = {
        "description": """
    Ratio of the MLP hidden dim to the embedding dim.
    """,
    }

    qkv_bias = {
        "description": """
    Whether to add a bias to the queries, keys and values.
    """,
    }

    n_embd = {
        "description": """
    Dimensionality of the embeddings and hidden states.
    """,
    }

    resid_pdrop = {
        "description": """
    The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    """,
    }

    embd_pdrop = {
        "description": """
    The dropout ratio for the embeddings.
    """,
    }

    clip_qkv = {
        "description": """
    If not `None`, cap the absolute value of the query, key, and value tensors to this value.
    """,
    }

    type_vocab_size = {
        "description": """
    The vocabulary size of the `token_type_ids`.
    """,
    }

    audio_config = {
        "description": """
    The config object or dictionary of the audio backbone.
    """,
    }

    layerdrop = {
        "description": """
    The LayerDrop probability. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for
    more details.
    """,
    }

    expert_capacity = {
        "description": """
    The number of tokens that each expert can process. If `None`, `expert_capacity` will be set to
    `(sequence_length / num_experts) * capacity_factor`.
    """,
    }

    decoder_start_token_id = {
        "description": """
    If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
    """,
    }

    is_encoder_decoder = {
        "description": """
    Whether the model is used as an encoder/decoder or not.
    """,
    }

    num_codebooks = {
        "description": """
    The number of parallel codebooks used by the model.
    """,
    }

    codebook_dim = {
        "description": """
    Dimensionality of each codebook embedding vector.
    """,
    }

    hidden_sizes = {
        "description": """
    Dimensionality (hidden size) at each stage of the model.
    """,
    }

    depths = {
        "description": """
    Depth of each layer in the Transformer.
    """,
    }

    patch_sizes = {
        "description": """
    Patch size at each stage of the model.
    """,
    }

    strides = {
        "description": """
    Stride at each stage of the model.
    """,
    }

    router_jitter_noise = {
        "description": """
    Amount of noise to add to the router logits during training for better load balancing.
    """,
    }

    num_local_experts = {
        "description": """
    Number of local experts on each device. `num_experts` should be divisible by `num_local_experts`.
    """,
    }

    qk_layernorm = {
        "description": """
    Whether to use query-key normalization in the attention.
    """,
    }

    backbone_config = {
        "description": """
    The configuration of the backbone model.
    """,
    }

    no_object_weight = {
        "description": """
    Relative classification weight of the no-object class in the object detection loss.
    """,
    }

    class_weight = {
        "description": """
    Relative weight of the classification error in the Hungarian matching cost.
    """,
    }

    mask_weight = {
        "description": """
    Relative weight of the focal loss in the panoptic segmentation loss.
    """,
    }

    dice_weight = {
        "description": """
    Relative weight of the dice loss in the panoptic segmentation loss.
    """,
    }

    class_cost = {
        "description": """
    Relative weight of the classification error in the Hungarian matching cost.
    """,
    }

    bbox_cost = {
        "description": """
    Relative weight of the L1 bounding box error in the Hungarian matching cost.
    """,
    }

    giou_cost = {
        "description": """
    Relative weight of the generalized IoU loss in the Hungarian matching cost.
    """,
    }

    focal_alpha = {
        "description": """
    Alpha parameter in the focal loss.
    """,
    }

    mask_loss_coefficient = {
        "description": """
    Relative weight of the focal loss in the panoptic segmentation loss.
    """,
    }

    giou_loss_coefficient = {
        "description": """
    Relative weight of the generalized IoU loss in the panoptic segmentation loss.
    """,
    }

    bbox_loss_coefficient = {
        "description": """
    Relative weight of the L1 bounding box loss in the panoptic segmentation loss.
    """,
    }

    cls_loss_coefficient = {
        "description": """
    Relative weight of the classification loss in the panoptic segmentation loss.
    """,
    }

    dice_loss_coefficient = {
        "description": """
    Relative weight of the dice loss in the panoptic segmentation loss.
    """,
    }

    semantic_loss_ignore_index = {
        "description": """
    The index that is ignored by the loss function of the semantic segmentation model.
    """,
    }

    projection_dim = {
        "description": """
    Dimensionality of text and vision projection layers.
    """,
    }

    logit_scale_init_value = {
        "description": """
    The initial value of the *logit_scale* parameter.
    """,
    }

    num_dense_layers = {
        "description": """
    Number of initial dense layers before MoE layers begin. Layers with index < num_dense_layers will use
    standard dense MLPs instead of MoE.
    """,
    }

    drop_path_rate = {
        "description": """
    Drop path rate for the patch fusion.
    """,
    }

    vq_config = {
        "description": """
    Configuration dict of the vector quantize module.
    """,
    }

    num_embeddings = {
        "description": """
    Number of codebook embeddings.
    """,
    }

    double_latent = {
        "description": """
    Whether to use double z channels.
    """,
    }

    latent_channels = {
        "description": """
    Number of channels for the latent space.
    """,
    }

    qformer_config = {
        "description": """
    Configuration dict of the Q-Former module.
    """,
    }

    conv_kernel_size = {
        "description": """
    The size of the convolutional kernel.
    """,
    }

    output_stride = {
        "description": """
    The ratio between the spatial resolution of the input and output feature maps.
    """,
    }

    depth_multiplier = {
        "description": """
    Shrinks or expands the number of channels in each layer. This is sometimes also called "alpha" or "width multiplier".
    """,
    }

    use_absolute_position_embeddings = {
        "description": """
    Whether to use absolute position embeddings.
    """,
    }

    use_relative_position_bias = {
        "description": """
    Whether to use relative position bias in the self-attention layers.
    """,
    }

    layer_scale_init_value = {
        "description": """
    Scale to use in the self-attention layers. 0.1 for base, 1e-6 for large. Set 0 to disable layer scale.
    """,
    }

    vlm_config = {
        "description": """
    The config object or dictionary of the vision-language backbone.
    """,
    }

    init_xavier_std = {
        "description": """
    The scaling factor used for the Xavier initialization of the cross-attention weights.
    """,
    }

    auxiliary_loss = {
        "description": """
    Whether auxiliary decoding losses (losses at each decoder layer) are to be used.
    """,
    }

    encoder_config = {
        "description": """
    The config object or dictionary of the encoder backbone.
    """,
    }

    decoder_config = {
        "description": """
    The config object or dictionary of the decoder backbone.
    """,
    }

    embedding_multiplier = {
        "description": """
    Scaling factor applied to the word embeddings. Used to scale the embeddings relative to the hidden size.
    """,
    }

    logits_scaling = {
        "description": """
    Scaling factor applied to the output logits before computing the probability distribution.
    """,
    }

    residual_multiplier = {
        "description": """
    Scaling factor applied to the residual connections.
    """,
    }

    attention_multiplier = {
        "description": """
    Scaling factor applied to the attention weights.
    """,
    }

    classifier_activation = {
        "description": """
    The activation function for the classification head.
    """,
    }

    return_dict = {
        "description": """
    Whether to return a `ModelOutput` (dataclass) instead of a plain tuple.
    """,
    }

    router_z_loss_coef = {
        "description": """
    Coefficient for the router z-loss, which penalizes large router logits to improve training stability.
    """,
    }

    final_logit_softcapping = {
        "description": """
    Soft-capping value applied to the final logits before computing the probability distribution. Logits are
    scaled by `tanh(logit / cap) * cap`.
    """,
    }

    cross_attention_hidden_size = {
        "description": """
    Hidden size of the encoder outputs projected into the cross-attention key/value space of the decoder. Used
    when the encoder and decoder have different hidden sizes.
    """,
    }

    input_dim = {
        "description": """
    Dimensionality of the input acoustic features (e.g., number of mel-filterbank channels).
    """,
    }

    use_auxiliary_loss = {
        "description": """
    Whether to calculate loss using intermediate predictions from transformer decoder.
    """,
    }

    batch_norm_eps = {
        "description": """
    The epsilon used by the batch normalization layers.
    """,
    }

    max_window_layers = {
        "description": """
    The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
    additional layer afterwards will use SWA (Sliding Window Attention).
    """,
    }

    ctc_loss_reduction = {
        "description": """
    Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training.
    """,
    }

    mask_feature_prob = {
        "description": """
    Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
    masking procedure generates `mask_feature_prob*len(feature_axis)/mask_time_length` independent masks over
    the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector
    span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
    may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment` is
    `True`.
    """,
    }

    eos_coefficient = {
        "description": """
    Relative classification weight of the 'no-object' class in the object detection loss.
    """,
    }

    num_labels = {
        "description": """
    Number of labels to use in the last layer added to the model, typically for a classification task.
    """,
    }

    depth = {
        "description": """
    Number of Transformer layers in the vision encoder.
    """,
    }

    temporal_patch_size = {
        "description": """
    Temporal patch size used in the 3D patch embedding for video inputs.
    """,
    }

    spatial_merge_size = {
        "description": """
        The size of the spatial merge window used to reduce the number of visual tokens by merging neighboring patches.
    """,
    }

    vision_start_token_id = {
        "description": """
    Token ID that marks the start of a visual segment in the multimodal input sequence.
    """,
    }

    vision_end_token_id = {
        "description": """
    Token ID that marks the end of a visual segment in the multimodal input sequence.
    """,
    }

    mamba_n_heads = {
        "description": """
    The number of mamba heads used in the v2 implementation.
    """,
    }

    mamba_d_head = {
        "description": """
    Head embedding dimension size
    """,
    }

    mamba_n_groups = {
        "description": """
    The number of the mamba groups used in the v2 implementation.
    """,
    }

    mamba_d_conv = {
        "description": """
    The size of the mamba convolution kernel
    """,
    }

    mamba_expand = {
        "description": """
    Expanding factor (relative to hidden_size) used to determine the mamba intermediate size
    """,
    }

    mamba_chunk_size = {
        "description": """
    The chunks in which to break the sequence when doing prefill/training
    """,
    }

    mamba_conv_bias = {
        "description": """
    Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
    """,
    }

    mamba_proj_bias = {
        "description": """
    Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block
    """,
    }

    time_step_min = {
        "description": """
    Minimum `time_step` used to bound `dt_proj.bias`.
    """,
    }

    time_step_max = {
        "description": """
    Maximum `time_step` used to bound `dt_proj.bias`.
    """,
    }

    time_step_limit = {
        "description": """
    Accepted range of time step values for clamping.
    """,
    }

    expand_ratio = {
        "description": """
    Expand ratio to set the output dimensions for the expansion
    """,
    }

    state_size = {
        "description": """
    Size of the SSM state (latent state dimension) in the Mamba layers.
    """,
    }

    time_step_rank = {
        "description": """
    Rank of the delta (time step) projection. Can be `"auto"` to set it automatically.
    """,
    }

    time_step_floor = {
        "description": """
    Minimum allowed value for the discrete time step delta after softplus activation.
    """,
    }

    time_step_scale = {
        "description": """
    Scale applied to the time step delta before discretization.
    """,
    }

    time_step_init_scheme = {
        "description": """
    Initialization scheme for the time step delta. Can be `"random"` or `"uniform"`.
    """,
    }

    mamba_d_ssm = {
        "description": """
    Inner state size of the SSM (state-space model) in the Mamba layers of FalconH1.
    """,
    }

    mamba_norm_before_gate = {
        "description": """
    Whether to apply normalization before the gating mechanism in the Mamba mixer.
    """,
    }

    mamba_rms_norm = {
        "description": """
    Whether to use RMS normalization in the Mamba layers (as opposed to standard LayerNorm).
    """,
    }

    mamba_d_state = state_size
    mamba_num_heads = mamba_n_heads
    mamba_head_dim = mamba_d_head
    num_input_channels = num_channels
    audio_channels = num_channels
    input_channels = num_channels
    in_channels = num_channels
    in_chans = num_channels
    scale_attn_weights = scale_embedding
    attention_probs_dropout_prob = attention_dropout
    attn_pdrop = attention_dropout
    attn_dropout = attention_dropout
    dropout = dropout_rate
    resid_dropout = resid_pdrop
    residual_dropout = resid_pdrop
    emb_pdrop = embd_pdrop
    embed_dropout = embd_pdrop
    embedding_dropout = embd_pdrop
    hidden_dropout_prob = hidden_dropout
    hidden_dropout_rate = hidden_dropout
    classifier_dropout_prob = classifier_dropout
    classifier_dropout_rate = classifier_dropout
    dropout_prob = dropout
    dropout_p = dropout
    decoder_attention_dropout = attention_dropout
    decoder_dropout = dropout
    encoder_dropout = dropout

    route_scale = routed_scaling_factor
    activation_function = hidden_act
    hidden_dim = hidden_size
    num_decoder_attention_heads = decoder_attention_heads
    num_encoder_attention_heads = encoder_attention_heads
    decoder_num_heads = decoder_attention_heads
    decoder_num_attention_heads = decoder_attention_heads
    encoder_num_heads = encoder_attention_heads
    encoder_num_attention_heads = encoder_attention_heads
    encoder_layers = num_encoder_layers
    decoder_layers = num_decoder_layers
    decoder_num_layers = num_decoder_layers
    encoder_num_layers = num_encoder_layers
    d_ff = intermediate_size
    dim_ff = intermediate_size
    n_inner = intermediate_size
    decoder_intermediate_size = intermediate_size
    num_kv_heads = num_key_value_heads
    num_layers = num_hidden_layers
    n_layers = num_hidden_layers
    n_layer = num_hidden_layers
    layers = num_layers
    encoder_num_hidden_layers = encoder_layers
    decoder_num_hidden_layers = decoder_layers
    num_heads = num_attention_heads
    n_heads = num_attention_heads
    n_head = num_attention_heads
    hidden_activation = hidden_act
    activation = hidden_act
    mlp_hidden_act = hidden_act
    d_head = head_dim
    d_inner = intermediate_size
    dim_head = head_dim
    ffn_dim = intermediate_size
    attention_heads = num_attention_heads
    n_positions = max_position_embeddings
    init_std = initializer_range
    initializer_std = initializer_range
    projector_bias = multimodal_projector_bias
    image_token_index = image_token_id
    video_token_index = video_token_id
    audio_token_index = audio_token_id
    embedding_size = n_embd
    embed_dim = n_embd
    projection_hidden_act = projector_hidden_act
    layer_norm_epsilon = layer_norm_eps
    rms_norm = rms_norm_eps
    norm_eps = layer_norm_eps
    eps = layer_norm_eps
    norm_epsilon = layer_norm_eps
    qk_layernorms = qk_layernorm
    use_qk_norm = qk_layernorm
    use_qkv_bias = qkv_bias

    decoder_hidden_act = hidden_act
    decoder_hidden_dim = hidden_size
    decoder_hidden_size = hidden_size
    encoder_hidden_dim = hidden_size
    encoder_hidden_size = hidden_size
    layer_scale_initial_scale = layer_scale_init_value
    multi_modal_projector_bias = projector_bias
    projector_hidden_size = projection_dim
    projection_size = projection_dim
    kernel_size = conv_kernel_size
    conv_kernel = conv_kernel_size
    use_absolute_embeddings = use_absolute_position_embeddings
    use_abs_pos = use_absolute_position_embeddings
    use_rel_pos = use_relative_position_bias
    aux_loss_coef = router_aux_loss_coef
    embedding_dimension = embed_dim
    embedding_dim = embed_dim
    emb_dim = embed_dim
    n_codebooks = num_codebooks
    codebook_size = num_codebooks
    layers_block_type = layer_types
    sample_rate = sampling_rate
    text_vocab_size = vocab_size


class ModelArgs:
    labels = {
        "description": """
    Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
    config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
    (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    """,
        "shape": "of shape `(batch_size, sequence_length)`",
    }

    num_logits_to_keep = {
        "description": """
    Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
    `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
    token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
    """,
        "shape": None,
    }

    input_ids = {
        "description": """
    Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

    Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
    [`PreTrainedTokenizer.__call__`] for details.

    [What are input IDs?](../glossary#input-ids)
    """,
        "shape": "of shape `(batch_size, sequence_length)`",
    }

    input_values = {
        "description": """
    Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
    into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
    (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
    To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and conversion
    into a tensor of type `torch.FloatTensor`. See [`{processor_class}.__call__`] for details.
    """,
        "shape": "of shape `(batch_size, sequence_length)`",
    }

    attention_mask = {
        "description": """
    Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

    - 1 for tokens that are **not masked**,
    - 0 for tokens that are **masked**.

    [What are attention masks?](../glossary#attention-mask)
    """,
        "shape": "of shape `(batch_size, sequence_length)`",
    }

    decoder_attention_mask = {
        "description": """
    Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
    make sure the model can only look at previous inputs in order to predict the future.
    """,
        "shape": "of shape `(batch_size, target_sequence_length)`",
    }

    encoder_hidden_states = {
        "description": """
    Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
    if the model is configured as a decoder.
    """,
        "shape": "of shape `(batch_size, sequence_length, hidden_size)`",
    }

    encoder_attention_mask = {
        "description": """
    Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
    the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

    - 1 for tokens that are **not masked**,
    - 0 for tokens that are **masked**.
    """,
        "shape": "of shape `(batch_size, sequence_length)`",
    }

    token_type_ids = {
        "description": """
    Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

    - 0 corresponds to a *sentence A* token,
    - 1 corresponds to a *sentence B* token.

    [What are token type IDs?](../glossary#token-type-ids)
    """,
        "shape": "of shape `(batch_size, sequence_length)`",
    }

    mm_token_type_ids = {
        "description": """
    Indices of input sequence tokens matching each modality. For example text (0), image (1), video (2).
    Multimodal token type ids can be obtained using [`AutoProcessor`]. See [`ProcessorMixin.__call__`] for details.

    """,
        "shape": "of shape `(batch_size, sequence_length)`",
    }

    position_ids = {
        "description": """
    Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

    [What are position IDs?](../glossary#position-ids)
    """,
        "shape": "of shape `(batch_size, sequence_length)`",
    }

    past_key_values = {
        "description": """
    Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
    blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
    returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

    Only [`~cache_utils.Cache`] instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
    If no `past_key_values` are passed, [`~cache_utils.DynamicCache`] will be initialized by default.

    The model will output the same cache format that is fed as input.

    If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
    have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
    of shape `(batch_size, sequence_length)`.
    """,
        "shape": None,
    }

    inputs_embeds = {
        "description": """
    Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
    is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
    model's internal embedding lookup matrix.
    """,
        "shape": "of shape `(batch_size, sequence_length, hidden_size)`",
    }

    decoder_input_ids = {
        "description": """
    Indices of decoder input sequence tokens in the vocabulary.

    Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
    [`PreTrainedTokenizer.__call__`] for details.

    [What are decoder input IDs?](../glossary#decoder-input-ids)
    """,
        "shape": "of shape `(batch_size, target_sequence_length)`",
    }

    decoder_inputs_embeds = {
        "description": """
    Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
    representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
    input (see `past_key_values`). This is useful if you want more control over how to convert
    `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

    If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
    of `inputs_embeds`.
    """,
        "shape": "of shape `(batch_size, target_sequence_length, hidden_size)`",
    }

    use_cache = {
        "description": """
    If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
    `past_key_values`).
    """,
        "shape": None,
    }

    output_attentions = {
        "description": """
    Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
    tensors for more detail.
    """,
        "shape": None,
    }

    output_hidden_states = {
        "description": """
    Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
    more detail.
    """,
        "shape": None,
    }

    return_dict = {
        "description": """
    Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """,
        "shape": None,
    }

    hidden_states = {
        "description": """ input to the layer of shape `(batch, seq_len, embed_dim)""",
        "shape": None,
    }

    interpolate_pos_encoding = {
        "description": """
    Whether to interpolate the pre-trained position encodings.
    """,
        "shape": None,
    }

    position_embeddings = {
        "description": """
    Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
    with `head_dim` being the embedding dimension of each attention head.
    """,
        "shape": None,
    }

    config = {
        "description": """
    Model configuration class with all the parameters of the model. Initializing with a config file does not
    load the weights associated with the model, only the configuration. Check out the
    [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """,
        "shape": None,
    }

    start_positions = {
        "description": """
    Labels for position (index) of the start of the labelled span for computing the token classification loss.
    Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
    are not taken into account for computing the loss.
    """,
        "shape": "of shape `(batch_size,)`",
    }

    end_positions = {
        "description": """
    Labels for position (index) of the end of the labelled span for computing the token classification loss.
    Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
    are not taken into account for computing the loss.
    """,
        "shape": "of shape `(batch_size,)`",
    }

    encoder_outputs = {
        "description": """
    Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
    `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
    hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
    """,
        "shape": None,
    }

    output_router_logits = {
        "description": """
    Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
    should not be returned during inference.
    """,
        "shape": None,
    }

    logits_to_keep = {
        "description": """
    If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
    `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
    token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
    If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
    This is useful when using packed tensor format (single dimension for batch and sequence length).
    """,
        "shape": None,
    }

    pixel_values = {
        "description": """
    The tensors corresponding to the input images. Pixel values can be obtained using
    [`{image_processor_class}`]. See [`{image_processor_class}.__call__`] for details ([`{processor_class}`] uses
    [`{image_processor_class}`] for processing images).
    """,
        "shape": "of shape `(batch_size, num_channels, image_size, image_size)`",
    }

    pixel_values_videos = {
        "description": """
    The tensors corresponding to the input video. Pixel values for videos can be obtained using
    [`{video_processor_class}`]. See [`{video_processor_class}.__call__`] for details ([`{processor_class}`] uses
    [`{video_processor_class}`] for processing videos).
    """,
        "shape": "of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`",
    }

    vision_feature_layer = {
        "description": """
    The index of the layer to select the vision feature. If multiple indices are provided,
    the vision feature of the corresponding indices will be concatenated to form the
    vision features.
    """,
        "shape": None,
    }

    vision_feature_select_strategy = {
        "description": """
    The feature selection strategy used to select the vision feature from the vision backbone.
    Can be one of `"default"` or `"full"`.
    """,
        "shape": None,
    }

    image_sizes = {
        "description": """
    The sizes of the images in the batch, being (height, width) for each image.
    """,
        "shape": "of shape `(batch_size, 2)`",
    }

    pixel_mask = {
        "description": """
    Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

    - 1 for pixels that are real (i.e. **not masked**),
    - 0 for pixels that are padding (i.e. **masked**).

    [What are attention masks?](../glossary#attention-mask)
    """,
        "shape": "of shape `(batch_size, height, width)`",
    }

    input_features = {
        "description": """
    The tensors corresponding to the input audio features. Audio features can be obtained using
    [`{feature_extractor_class}`]. See [`{feature_extractor_class}.__call__`] for details ([`{processor_class}`] uses
    [`{feature_extractor_class}`] for processing audios).
    """,
        "shape": "of shape `(batch_size, sequence_length, feature_dim)`",
    }


class ModelOutputArgs:
    last_hidden_state = {
        "description": """
    Sequence of hidden-states at the output of the last layer of the model.
    """,
        "shape": "of shape `(batch_size, sequence_length, hidden_size)`",
    }

    past_key_values = {
        "description": """
    It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

    Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
    `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
    input) to speed up sequential decoding.
    """,
        "shape": None,
        "additional_info": "returned when `use_cache=True` is passed or when `config.use_cache=True`",
    }

    hidden_states = {
        "description": """
    Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
    one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """,
        "shape": None,
        "additional_info": "returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`",
    }

    attentions = {
        "description": """
    Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
    sequence_length)`.

    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
    heads.
    """,
        "shape": None,
        "additional_info": "returned when `output_attentions=True` is passed or when `config.output_attentions=True`",
    }

    pooler_output = {
        "description": """
    Last layer hidden-state after a pooling operation on the spatial dimensions.
    """,
        "shape": "of shape `(batch_size, hidden_size)`",
    }

    cross_attentions = {
        "description": """
    Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
    sequence_length)`.

    Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
    weighted average in the cross-attention heads.
    """,
        "shape": None,
        "additional_info": "returned when `output_attentions=True` is passed or when `config.output_attentions=True`",
    }

    decoder_hidden_states = {
        "description": """
    Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
    one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

    Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    """,
        "shape": None,
        "additional_info": "returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`",
    }

    decoder_attentions = {
        "description": """
    Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
    sequence_length)`.

    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
    self-attention heads.
    """,
        "shape": None,
        "additional_info": "returned when `output_attentions=True` is passed or when `config.output_attentions=True`",
    }

    encoder_last_hidden_state = {
        "description": """
    Sequence of hidden-states at the output of the last layer of the encoder of the model.
    """,
        "shape": "of shape `(batch_size, sequence_length, hidden_size)`",
    }

    encoder_hidden_states = {
        "description": """
    Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
    one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

    Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    """,
        "shape": None,
        "additional_info": "returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`",
    }

    encoder_attentions = {
        "description": """
    Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
    sequence_length)`.

    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
    self-attention heads.
    """,
        "shape": None,
        "additional_info": "returned when `output_attentions=True` is passed or when `config.output_attentions=True`",
    }

    router_logits = {
        "description": """
    Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

    Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.
    """,
        "shape": None,
        "additional_info": "returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`",
    }

    router_probs = {
        "description": """
    Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

    Raw router probabilities that are computed by MoE routers, these terms are used to compute the auxiliary
    loss and the z_loss for Mixture of Experts models.
    """,
        "shape": None,
        "additional_info": "returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`",
    }

    z_loss = {
        "description": """
    z_loss for the sparse modules.
    """,
        "shape": None,
        "additional_info": "returned when `labels` is provided",
    }

    aux_loss = {
        "description": """
    aux_loss for the sparse modules.
    """,
        "shape": None,
        "additional_info": "returned when `labels` is provided",
    }

    start_logits = {
        "description": """
    Span-start scores (before SoftMax).
    """,
        "shape": "of shape `(batch_size, sequence_length)`",
    }

    end_logits = {
        "description": """
    Span-end scores (before SoftMax).
    """,
        "shape": "of shape `(batch_size, sequence_length)`",
    }

    feature_maps = {
        "description": """
    Feature maps of the stages.
    """,
        "shape": "of shape `(batch_size, num_channels, height, width)`",
    }

    reconstruction = {
        "description": """
    Reconstructed / completed images.
    """,
        "shape": "of shape `(batch_size, num_channels, height, width)`",
    }

    spectrogram = {
        "description": """
    The predicted spectrogram.
    """,
        "shape": "of shape `(batch_size, sequence_length, num_bins)`",
    }

    predicted_depth = {
        "description": """
    Predicted depth for each pixel.
    """,
        "shape": "of shape `(batch_size, height, width)`",
    }

    sequences = {
        "description": """
    Sampled values from the chosen distribution.
    """,
        "shape": "of shape `(batch_size, num_samples, prediction_length)` or `(batch_size, num_samples, prediction_length, input_size)`",
    }

    params = {
        "description": """
    Parameters of the chosen distribution.
    """,
        "shape": "of shape `(batch_size, num_samples, num_params)`",
    }

    loc = {
        "description": """
    Shift values of each time series' context window which is used to give the model inputs of the same
    magnitude and then used to shift back to the original magnitude.
    """,
        "shape": "of shape `(batch_size,)` or `(batch_size, input_size)`",
    }

    scale = {
        "description": """
    Scaling values of each time series' context window which is used to give the model inputs of the same
    magnitude and then used to rescale back to the original magnitude.
    """,
        "shape": "of shape `(batch_size,)` or `(batch_size, input_size)`",
    }

    static_features = {
        "description": """
    Static features of each time series' in a batch which are copied to the covariates at inference time.
    """,
        "shape": "of shape `(batch_size, feature size)`",
    }

    embeddings = {
        "description": """
    Utterance embeddings used for vector similarity-based retrieval.
    """,
        "shape": "of shape `(batch_size, config.xvector_output_dim)`",
    }

    extract_features = {
        "description": """
    Sequence of extracted feature vectors of the last convolutional layer of the model.
    """,
        "shape": "of shape `(batch_size, sequence_length, conv_dim[-1])`",
    }

    projection_state = {
        "description": """
    Text embeddings before the projection layer, used to mimic the last hidden state of the teacher encoder.
    """,
        "shape": "of shape `(batch_size,config.project_dim)`",
    }

    image_hidden_states = {
        "description": """
    Image hidden states of the model produced by the vision encoder and after projecting the last hidden state.
    """,
        "shape": "of shape `(batch_size, num_images, sequence_length, hidden_size)`",
    }

    video_hidden_states = {
        "description": """
    Video hidden states of the model produced by the vision encoder and after projecting the last hidden state.
    """,
        "shape": "of shape `(batch_size * num_frames, num_images, sequence_length, hidden_size)`",
    }


class ClassDocstring:
    Config = r"""
    This is the configuration class to store the configuration of a {model_base_class}. It is used to instantiate a {model_name}
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [{model_checkpoint}](https://huggingface.co/{model_checkpoint})

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    """

    PreTrainedModel = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    """

    Model = r"""
    The bare {model_name} Model outputting raw hidden-states without any specific head on top.
    """

    ForPreTraining = r"""
    The {model_name} Model with a specified pretraining head on top.
    """

    Decoder = r"""
    The bare {model_name} Decoder outputting raw hidden-states without any specific head on top.
    """

    TextModel = r"""
    The bare {model_name} Text Model outputting raw hidden-states without any specific head on to.
    """

    ForSequenceClassification = r"""
    The {model_name} Model with a sequence classification/regression head on top e.g. for GLUE tasks.
    """

    ForQuestionAnswering = r"""
    The {model_name} transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """

    ForMultipleChoice = r"""
    The {model_name} Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """

    ForMaskedLM = r"""
    The {model_name} Model with a `language modeling` head on top."
    """

    ForTokenClassification = r"""
    The {model_name} transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """

    ForConditionalGeneration = r"""
    The {model_name} Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).
    """

    ForCausalLM = r"""
    The {model_name} Model for causal language modeling.
    """

    Backbone = r"""
    The {model_name} backbone.
    """

    ForImageClassification = r"""
    The {model_name} Model with an image classification head on top e.g. for ImageNet.
    """
    ForSemanticSegmentation = r"""
    The {model_name} Model with a semantic segmentation head on top e.g. for ADE20K, CityScapes.
    """
    ForAudioClassification = r"""
    The {model_name} Model with an audio classification head on top (a linear layer on top of the pooled
    output).
    """

    ForAudioFrameClassification = r"""
    The {model_name} Model with a frame classification head on top for tasks like Speaker Diarization.
    """

    ForPrediction = r"""
    The {model_name} Model with a distribution head on top for time-series forecasting.
    """

    WithProjection = r"""
    The {model_name} Model with a projection layer on top (a linear layer on top of the pooled output).
    """


class ClassAttrs:
    # fmt: off
    base_model_prefix = r"""
    A string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
    """
    supports_gradient_checkpointing = r"""
    Whether the model supports gradient checkpointing or not. Gradient checkpointing is a memory-saving technique that trades compute for memory, by storing only a subset of activations (checkpoints) and recomputing the activations that are not stored during the backward pass.
    """
    _no_split_modules = r"""
    Layers of modules that should not be split across devices should be added to `_no_split_modules`. This can be useful for modules that contains skip connections or other operations that are not compatible with splitting the module across devices. Setting this attribute will enable the use of `device_map="auto"` in the `from_pretrained` method.
    """
    _skip_keys_device_placement = r"""
    A list of keys to ignore when moving inputs or outputs between devices when using the `accelerate` library.
    """
    _supports_flash_attn = r"""
    Whether the model's attention implementation supports FlashAttention.
    """
    _supports_sdpa = r"""
    Whether the model's attention implementation supports SDPA (Scaled Dot Product Attention).
    """
    _supports_flex_attn = r"""
    Whether the model's attention implementation supports FlexAttention.
    """
    _can_compile_fullgraph = r"""
    Whether the model can `torch.compile` fullgraph without graph breaks. Models will auto-compile if this flag is set to `True`
    in inference, if a compilable cache is used.
    """
    _supports_attention_backend = r"""
    Whether the model supports attention interface functions. This flag signal that the model can be used as an efficient backend in TGI and vLLM.
    """
    _tied_weights_keys = r"""
    A list of `state_dict` keys that are potentially tied to another key in the state_dict.
    """
    # fmt: on


ARGS_TO_IGNORE = {"self", "kwargs", "args", "deprecated_arguments"}
ARGS_TO_RENAME = {"_out_features": "out_features", "_out_indices": "out_indices"}


def get_indent_level(func):
    # Use this instead of `inspect.getsource(func)` as getsource can be very slow
    return (len(func.__qualname__.split(".")) - 1) * 4


def equalize_indent(docstring: str, indent_level: int) -> str:
    """
    Adjust the indentation of a docstring to match the specified indent level.
    """
    prefix = " " * indent_level
    # Uses splitlines() (no keepends) to match previous behaviour that dropped
    # any trailing newline via the old splitlines() + "\n".join() + textwrap.indent path.
    return "\n".join(prefix + line.lstrip() if line.strip() else "" for line in docstring.splitlines())


def set_min_indent(docstring: str, indent_level: int) -> str:
    """
    Adjust the indentation of a docstring to match the specified indent level.
    """
    # Equivalent to textwrap.dedent + textwrap.indent but avoids the two regex
    # passes that textwrap uses internally (one per call in dedent, one in indent).
    lines = docstring.split("\n")
    min_indent = min(
        (len(line) - len(line.lstrip()) for line in lines if line.strip()),
        default=0,
    )
    prefix = " " * indent_level
    return "\n".join(prefix + line[min_indent:] if line.strip() else "" for line in lines)


def parse_shape(docstring):
    match = _re_shape.search(docstring)
    if match:
        return " " + match.group(1)
    return None


def parse_default(docstring):
    match = _re_default.search(docstring)
    if match:
        return " " + match.group(1)
    return None


def parse_docstring(docstring, max_indent_level=0, return_intro=False):
    """
    Parse the docstring to extract the Args section and return it as a dictionary.
    The docstring is expected to be in the format:
    Args:
        arg1 (type):
            Description of arg1.
        arg2 (type):
            Description of arg2.

    # This function will also return the remaining part of the docstring after the Args section.
    Returns:/Example:
    ...
    """
    match = _re_example_or_return.search(docstring)
    if match:
        remainder_docstring = docstring[match.start() :]
        docstring = docstring[: match.start()]
    else:
        remainder_docstring = ""

    args_match = _re_args_section.search(docstring)
    # still try to find args description in the docstring, if args are not preceded by "Args:"
    docstring_intro = None
    if args_match:
        docstring_intro = docstring[: args_match.start()]
        if docstring_intro.split("\n")[-1].strip() == '"""':
            docstring_intro = "\n".join(docstring_intro.split("\n")[:-1])
        if docstring_intro.split("\n")[0].strip() == 'r"""' or docstring_intro.split("\n")[0].strip() == '"""':
            docstring_intro = "\n".join(docstring_intro.split("\n")[1:])
        if docstring_intro.strip() == "":
            docstring_intro = None
    args_section = args_match.group(1).lstrip("\n") if args_match else docstring
    if args_section.split("\n")[-1].strip() == '"""':
        args_section = "\n".join(args_section.split("\n")[:-1])
    if args_section.split("\n")[0].strip() == 'r"""' or args_section.split("\n")[0].strip() == '"""':
        args_section = "\n".join(args_section.split("\n")[1:])
    args_section = set_min_indent(args_section, 0)
    params = {}
    if args_section:
        # Use the pre-compiled pattern (max_indent_level is always 0 at all call
        # sites; if a non-zero value is ever needed, compile a fresh pattern).
        if max_indent_level == 0:
            param_pattern = _re_param
        else:
            param_pattern = re.compile(
                # |--- Group 1 ---|| Group 2 ||- Group 3 -||---------- Group 4 ----------|
                rf"^\s{{0,{max_indent_level}}}(\w+)\s*\(\s*([^, \)]*)(\s*.*?)\s*\)\s*:\s*((?:(?!\n^\s{{0,{max_indent_level}}}\w+\s*\().)*)",
                re.DOTALL | re.MULTILINE,
            )
        for match in param_pattern.finditer(args_section):
            param_name = match.group(1)
            param_type = match.group(2)
            additional_info = match.group(3)
            optional = "optional" in additional_info
            shape = parse_shape(additional_info)
            default = parse_default(additional_info)
            param_description = match.group(4).strip()
            # indent the first line of param_description to 4 spaces:
            param_description = " " * 4 + param_description
            param_description = f"\n{param_description}"
            params[param_name] = {
                "type": param_type,
                "description": param_description,
                "optional": optional,
                "shape": shape,
                "default": default,
                "additional_info": additional_info,
            }

    if params and remainder_docstring:
        remainder_docstring = "\n" + remainder_docstring

    remainder_docstring = set_min_indent(remainder_docstring, 0)

    if return_intro:
        return params, remainder_docstring, docstring_intro
    return params, remainder_docstring


def contains_type(type_hint, target_type) -> tuple[bool, object | None]:
    """
    Check if a "nested" type hint contains a specific target type,
    return the first-level type containing the target_type if found.
    """
    args = get_args(type_hint)
    if args == ():
        try:
            return issubclass(type_hint, target_type), type_hint
        except Exception:
            return issubclass(type(type_hint), target_type), type_hint
    found_type_tuple = [contains_type(arg, target_type)[0] for arg in args]
    found_type = any(found_type_tuple)
    if found_type:
        type_hint = args[found_type_tuple.index(True)]
    return found_type, type_hint


def get_model_name(obj):
    """
    Get the model name from the file path of the object.
    """

    path = inspect.getsourcefile(obj)
    if path is None:
        return None
    if path.split(os.path.sep)[-3] != "models":
        return None
    file_name = path.split(os.path.sep)[-1]
    model_name_lowercase_from_folder = path.split(os.path.sep)[-2]
    model_name_lowercase_from_file = None
    for file_type in AUTODOC_FILES:
        start = file_type.split("*")[0]
        end = file_type.split("*")[-1] if "*" in file_type else ""
        if file_name.startswith(start) and file_name.endswith(end):
            model_name_lowercase_from_file = file_name[len(start) : -len(end)]
            break
    if model_name_lowercase_from_file and model_name_lowercase_from_folder != model_name_lowercase_from_file:
        from transformers.models.auto.configuration_auto import SPECIAL_MODEL_TYPE_TO_MODULE_NAME

        if (
            model_name_lowercase_from_file in SPECIAL_MODEL_TYPE_TO_MODULE_NAME
            or model_name_lowercase_from_file.replace("_", "-") in SPECIAL_MODEL_TYPE_TO_MODULE_NAME
        ):
            return model_name_lowercase_from_file
        return model_name_lowercase_from_folder
    return model_name_lowercase_from_folder


def generate_processor_intro(cls) -> str:
    """
    Generate the intro docstring for a processor class based on its attributes.

    Args:
        cls: Processor class to generate intro for

    Returns:
        str: Generated intro text
    """
    class_name = cls.__name__

    # Get attributes and their corresponding class names
    attributes = cls.get_attributes()
    if not attributes:
        return ""

    # Build list of component names and their classes
    components = []
    component_classes = []

    for attr in attributes:
        # Get the class name for this attribute
        class_attr = f"{attr}_class"
        # Format attribute name for display
        attr_display = attr.replace("_", " ")
        components.append(attr_display)
        component_classes.append(f"[`{{{class_attr}}}`]")
    if not components:
        return ""

    # Generate the intro text
    if len(components) == 1:
        components_text = f"a {components[0]}"
        classes_text = component_classes[0]
        classes_text_short = component_classes[0].replace("[`", "[`~")
    elif len(components) == 2:
        components_text = f"a {components[0]} and a {components[1]}"
        classes_text = f"{component_classes[0]} and {component_classes[1]}"
        classes_text_short = (
            f"{component_classes[0].replace('[`', '[`~')} and {component_classes[1].replace('[`', '[`~')}"
        )
    else:
        components_text = ", ".join(f"a {c}" for c in components[:-1]) + f", and a {components[-1]}"
        classes_text = ", ".join(component_classes[:-1]) + f", and {component_classes[-1]}"
        classes_short = [c.replace("[`", "[`~") for c in component_classes]
        classes_text_short = ", ".join(classes_short[:-1]) + f", and {classes_short[-1]}"

    intro = f"""Constructs a {class_name} which wraps {components_text} into a single processor.

[`{class_name}`] offers all the functionalities of {classes_text}. See the
{classes_text_short} for more information.
"""

    return intro


def get_placeholders_dict(placeholders: set[str], model_name: str) -> Mapping[str, str | None]:
    """
    Get the dictionary of placeholders for the given model name.
    """
    # import here to avoid circular import
    from transformers.models import auto as auto_module

    placeholders_dict = {}
    for placeholder in placeholders:
        # Infer placeholders from the model name and the auto modules
        if placeholder in PLACEHOLDER_TO_AUTO_MODULE:
            try:
                place_holder_value = getattr(
                    getattr(auto_module, PLACEHOLDER_TO_AUTO_MODULE[placeholder][0]),
                    PLACEHOLDER_TO_AUTO_MODULE[placeholder][1],
                ).get(model_name, None)
            except ImportError:
                # In case a library is not installed, we don't want to fail the docstring generation
                place_holder_value = None
            if place_holder_value is not None:
                if isinstance(place_holder_value, list | tuple):
                    place_holder_value = (
                        place_holder_value[-1] if place_holder_value[-1] is not None else place_holder_value[0]
                    )
                placeholders_dict[placeholder] = place_holder_value if place_holder_value is not None else placeholder
            else:
                placeholders_dict[placeholder] = placeholder

    return placeholders_dict


def format_args_docstring(docstring: str, model_name: str) -> str:
    """
    Replaces placeholders such as {image_processor_class} in the docstring with the actual values,
    deducted from the model name and the auto modules.
    """
    # first check if there are any placeholders in the docstring, if not return it as is
    placeholders = set(_re_placeholders.findall(docstring))
    if not placeholders:
        return docstring

    # get the placeholders dictionary for the given model name
    placeholders_dict = get_placeholders_dict(placeholders, model_name)
    # replace the placeholders in the docstring with the values from the placeholders_dict
    for placeholder, value in placeholders_dict.items():
        if isinstance(value, dict) and placeholder == "image_processor_class":
            value = value.get("torchvision", value.get("pil", None))
        if placeholder is not None:
            docstring = docstring.replace(f"{{{placeholder}}}", value)
    return docstring


def get_args_doc_from_source(args_classes: object | list[object]) -> dict:
    if isinstance(args_classes, list | tuple):
        return _merge_args_dicts(tuple(args_classes))
    return args_classes.__dict__


@lru_cache(maxsize=16)
def _merge_args_dicts(args_classes_tuple: tuple) -> dict:
    """Cached merger of args-doc dicts. The input classes are static so caching is safe."""
    result = {}
    for cls in args_classes_tuple:
        result.update(cls.__dict__)
    return result


def get_checkpoint_from_config_class(config_class):
    checkpoint = None

    # source code of `config_class`
    # config_source = inspect.getsource(config_class)
    config_source = config_class.__doc__
    if not config_source:
        return None

    checkpoints = _re_checkpoint.findall(config_source)
    # Each `checkpoint` is a tuple of a checkpoint name and a checkpoint link.
    # For example, `('google-bert/bert-base-uncased', 'https://huggingface.co/google-bert/bert-base-uncased')`
    for ckpt_name, ckpt_link in checkpoints:
        # allow the link to end with `/`
        ckpt_link = ckpt_link.removesuffix("/")

        # verify the checkpoint name corresponds to the checkpoint link
        ckpt_link_from_name = f"https://huggingface.co/{ckpt_name}"
        if ckpt_link == ckpt_link_from_name:
            checkpoint = ckpt_name
            break

    return checkpoint


def add_intro_docstring(func, class_name, indent_level=0):
    intro_docstring = ""
    if func.__name__ == "forward":
        intro_docstring = rf"""The [`{class_name}`] forward method, overrides the `__call__` special method.

        <Tip>

        Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
        instance afterwards instead of this since the former takes care of running the pre and post processing steps while
        the latter silently ignores them.

        </Tip>

        """
        intro_docstring = equalize_indent(intro_docstring, indent_level + 4)

    return intro_docstring


def _get_model_info(func, parent_class):
    """
    Extract model information from a function or its parent class.

    Args:
        func (`function`): The function to extract information from
        parent_class (`class`): Optional parent class of the function
    """
    # import here to avoid circular import
    from transformers.models import auto as auto_module

    # Get model name from either parent class or function
    if parent_class is not None:
        model_name_lowercase = get_model_name(parent_class)
    else:
        model_name_lowercase = get_model_name(func)

    # Normalize model name if needed
    if model_name_lowercase and model_name_lowercase not in getattr(
        getattr(auto_module, PLACEHOLDER_TO_AUTO_MODULE["config_class"][0]),
        PLACEHOLDER_TO_AUTO_MODULE["config_class"][1],
    ):
        model_name_lowercase = model_name_lowercase.replace("_", "-")

    # Get class name from function's qualified name
    class_name = func.__qualname__.split(".")[0]

    # Get config class for the model
    if model_name_lowercase is None:
        config_class = None
    else:
        try:
            config_class = getattr(
                getattr(auto_module, PLACEHOLDER_TO_AUTO_MODULE["config_class"][0]),
                PLACEHOLDER_TO_AUTO_MODULE["config_class"][1],
            )[model_name_lowercase]
        except KeyError:
            if model_name_lowercase in HARDCODED_CONFIG_FOR_MODELS:
                config_class = HARDCODED_CONFIG_FOR_MODELS[model_name_lowercase]
            else:
                config_class = "ModelConfig"
                print(
                    f"[ERROR] Config not found for {model_name_lowercase}. You can manually add it to HARDCODED_CONFIG_FOR_MODELS in utils/auto_docstring.py"
                )
    return model_name_lowercase, class_name, config_class


def _format_type_annotation_recursive(type_hint):
    """
    Recursively format a type annotation object as a string, preserving generic type arguments.

    This is an internal helper used by process_type_annotation for the type object path.

    Args:
        type_hint: A type annotation object

    Returns:
        str: Formatted type string
    """
    # Handle special cases
    if type_hint is type(...) or type_hint is Ellipsis:
        return "..."
    # Note: NoneType handling is done later to preserve "NoneType" in Union[] but "None" in | syntax

    # Check if this is a generic type (e.g., list[str], dict[str, int])
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is not None and args:
        # This is a generic type - format it with its arguments
        # Get the origin type name
        if hasattr(origin, "__module__") and hasattr(origin, "__name__"):
            # Clean up module name - need to handle both 'typing.' prefix and just 'typing'
            module_name = origin.__module__
            if module_name in ("typing", "types", "builtins"):
                module_name = ""
            else:
                module_name = (
                    module_name.replace("transformers.", "~")
                    .replace("typing.", "")
                    .replace("types.", "")
                    .replace("builtins.", "")
                )

            if module_name:
                origin_str = f"{module_name}.{origin.__name__}"
            else:
                origin_str = origin.__name__
        else:
            origin_str = str(origin)

        # Handle special origin types
        if origin_str == "UnionType":
            # Python 3.13's X | Y syntax - format it nicely
            arg_strs = [_format_type_annotation_recursive(arg) for arg in args]
            return " | ".join(arg_strs)

        # Special handling for Annotated[Union[...], ...] and Annotated[UnionType[...], ...]
        # Check if first arg is a Union/UnionType and format it specially
        if origin_str == "Annotated" and args:
            first_arg_origin = get_origin(args[0])
            # Check if it's a UnionType (modern | syntax) or Union (old Union[] syntax)
            if first_arg_origin is UnionType:
                # Modern union type - format as X | Y | Z (with None not NoneType)
                union_args = get_args(args[0])
                union_strs = []
                for arg in union_args:
                    if arg is type(None):
                        union_strs.append("None")  # Modern syntax uses "None"
                    else:
                        union_strs.append(_format_type_annotation_recursive(arg))
                formatted_union = " | ".join(union_strs)
                # Include the rest of the Annotated metadata
                remaining_args = [_format_type_annotation_recursive(arg) for arg in args[1:]]
                all_args = [formatted_union] + remaining_args
                return f"{origin_str}[{', '.join(all_args)}]"
            elif first_arg_origin is Union:
                # Old-style Union - format as Union[X, Y, Z]
                union_args = get_args(args[0])
                union_strs = [_format_type_annotation_recursive(arg) for arg in union_args]
                formatted_union = f"Union[{', '.join(union_strs)}]"
                # Include the rest of the Annotated metadata
                remaining_args = [_format_type_annotation_recursive(arg) for arg in args[1:]]
                all_args = [formatted_union] + remaining_args
                return f"{origin_str}[{', '.join(all_args)}]"

        # Recursively format the generic arguments
        arg_strs = [_format_type_annotation_recursive(arg) for arg in args]
        return f"{origin_str}[{', '.join(arg_strs)}]"
    elif hasattr(type_hint, "__module__") and hasattr(type_hint, "__name__"):
        # Simple type with module and name
        # Clean up module name - need to handle both 'typing.' prefix and just 'typing'
        module_name = type_hint.__module__
        if module_name in ("typing", "types", "builtins"):
            module_name = ""
        else:
            module_name = (
                module_name.replace("transformers.", "~")
                .replace("typing.", "")
                .replace("types.", "")
                .replace("builtins.", "")
            )

        if module_name:
            type_name = f"{module_name}.{type_hint.__name__}"
        else:
            type_name = type_hint.__name__

        return type_name
    else:
        # Fallback to string representation
        type_str = str(type_hint)
        # Clean up ForwardRef
        if "ForwardRef" in type_str:
            type_str = _re_forward_ref.sub(r"\1", type_str)
        # Clean up module prefixes
        type_str = type_str.replace("typing.", "").replace("types.", "")
        return type_str


def process_type_annotation(type_input, param_name: str | None = None) -> tuple[str, bool]:
    """
    Unified function to process and format a parameter's type annotation.

    This function intelligently handles both type objects (from inspect.Parameter.annotation)
    and string representations of types. It will:
    - Use type introspection when given a type object (preserves generic arguments)
    - Parse string representations when that's all that's available
    - Always return a formatted type string and optional flag

    Handles various type representations including:
    - Type objects with generics (e.g., list[str], Optional[int])
    - Union types (both Union[X, Y] and X | Y syntax)
    - Modern union syntax with | (e.g., "bool | None")
    - Complex typing constructs (Union, Optional, Annotated, etc.)
    - Generic types with brackets
    - Class type strings
    - Simple types and module paths

    Args:
        type_input: Either a type annotation object or a string representation of a type
        param_name (`str | None`): The parameter name (used for legacy module path handling)

    Returns:
        tuple[str, bool]: (formatted_type_string, is_optional)
    """
    optional = False

    # Path 1: Type object (best approach - preserves generic type information)
    if not isinstance(type_input, str):
        # Handle None type
        if type_input is None or type_input is type(None):
            return "None", True

        # Handle Union types and modern UnionType (X | Y)
        if get_origin(type_input) is Union or get_origin(type_input) is UnionType:
            subtypes = get_args(type_input)
            out_str = []
            for subtype in subtypes:
                if subtype is type(None):
                    optional = True
                    continue
                formatted_type = _format_type_annotation_recursive(subtype)
                out_str.append(formatted_type)

            if not out_str:
                return "", optional
            elif len(out_str) == 1:
                return out_str[0], optional
            else:
                return f"Union[{', '.join(out_str)}]", optional

        # Single type (not a Union)
        formatted_type = _format_type_annotation_recursive(type_input)
        return formatted_type, optional

    # Path 2: String representation (fallback when we only have strings)
    param_type = type_input

    # Handle Union types with | syntax
    if " | " in param_type:
        # Modern union syntax (e.g., "bool | None")
        parts = [p.strip() for p in param_type.split(" | ")]
        if "None" in parts:
            optional = True
            parts = [p for p in parts if p != "None"]
        param_type = " | ".join(parts) if parts else ""
        # Clean up module prefixes including typing
        param_type = "".join(param_type.split("typing.")).replace("transformers.", "~").replace("builtins.", "")

    elif "typing" in param_type or "Union[" in param_type or "Optional[" in param_type or "[" in param_type:
        # Complex typing construct or generic type - clean up typing module references
        param_type = "".join(param_type.split("typing.")).replace("transformers.", "~")

    elif "<class '" in param_type:
        # This is a class type like "<class 'module.ClassName'>" - should NOT append param_name
        param_type = (
            param_type.replace("transformers.", "~").replace("builtins.", "").replace("<class '", "").replace("'>", "")
        )

    else:
        # Simple type or module path - only append param_name if it looks like a module path
        # This is legacy behavior for backwards compatibility
        if param_name and "." in param_type and not param_type.split(".")[-1][0].isupper():
            # Looks like a module path ending with an attribute
            param_type = f"{param_type.replace('transformers.', '~').replace('builtins', '')}.{param_name}"
        else:
            # Simple type name, don't append param_name
            param_type = param_type.replace("transformers.", "~").replace("builtins.", "")

    # Clean up ForwardRef
    if "ForwardRef" in param_type:
        param_type = _re_forward_ref.sub(r"\1", param_type)

    # Handle Optional wrapper
    if "Optional" in param_type:
        param_type = _re_optional.sub(r"\1", param_type)
        optional = True

    return param_type, optional


def _process_parameter_type(param):
    """
    Process and format a parameter's type annotation from an inspect.Parameter object.

    Args:
        param (`inspect.Parameter`): The parameter from the function signature

    Returns:
        tuple[str, bool]: (formatted_type_string, is_optional)
    """
    if param.annotation == inspect.Parameter.empty:
        return "", False

    # Use the unified function to process the type annotation
    formatted_type, optional = process_type_annotation(param.annotation)

    # Check if parameter has a default value (makes it optional)
    if param.default is not inspect.Parameter.empty:
        optional = True

    return formatted_type, optional


def _get_parameter_info(param_name, documented_params, source_args_dict, param_type, optional):
    """
    Get parameter documentation details from the appropriate source.
    Tensor shape, optional status and description are taken from the custom docstring in priority if available.
    Type is taken from the function signature first, then from the custom docstring if missing from the signature

    Args:
        param_name (`str`): Name of the parameter
        documented_params (`dict`): Dictionary of documented parameters (manually specified in the docstring)
        source_args_dict (`dict`): Default source args dictionary to use if not in documented_params
        param_type (`str`): Current parameter type (may be updated)
        optional (`bool`): Whether the parameter is optional (may be updated)
    """
    description = None
    shape = None
    shape_string = ""
    is_documented = True
    additional_info = None
    optional_string = r", *optional*" if optional else ""

    if param_name in documented_params:
        # Parameter is documented in the function's docstring
        if (
            param_type == ""
            and documented_params[param_name].get("type", None) is not None
            or documented_params[param_name]["additional_info"]
        ):
            param_type = documented_params[param_name]["type"]
        optional = documented_params[param_name]["optional"]
        shape = documented_params[param_name].get("shape", None)
        shape_string = shape if shape else ""
        additional_info = documented_params[param_name]["additional_info"] or ""
        description = f"{documented_params[param_name]['description']}\n"
    elif param_name in source_args_dict:
        # Parameter is documented in ModelArgs or ImageProcessorArgs
        param_type = source_args_dict[param_name].get("type", param_type)
        shape = source_args_dict[param_name].get("shape", None)
        shape_string = " " + shape if shape else ""
        description = source_args_dict[param_name]["description"]
        additional_info = source_args_dict[param_name].get("additional_info", None)
        if additional_info:
            additional_info = shape_string + optional_string + ", " + additional_info
    else:
        # Parameter is not documented
        is_documented = False

    return param_type, optional_string, shape_string, additional_info, description, is_documented


def _process_regular_parameters(
    sig,
    func,
    class_name,
    documented_params,
    indent_level,
    undocumented_parameters,
    source_args_dict,
    parent_class,
    allowed_params=None,
):
    """
    Process all regular parameters (not kwargs parameters) from the function signature.

    Args:
        sig (`inspect.Signature`): Function signature
        func (`function`): Function the parameters belong to
        class_name (`str`): Name of the class
        documented_params (`dict`): Dictionary of parameters that are already documented
        indent_level (`int`): Indentation level
        undocumented_parameters (`list`): List to append undocumented parameters to
    """
    docstring = ""
    # Check if this is a processor by inspecting class hierarchy
    is_processor = _is_processor_class(func, parent_class)

    # Use appropriate args source based on whether it's a processor or not
    if source_args_dict is None:
        if is_processor:
            source_args_dict = get_args_doc_from_source([ModelArgs, ImageProcessorArgs, ProcessorArgs])
        else:
            source_args_dict = get_args_doc_from_source([ModelArgs, ImageProcessorArgs])

    missing_args = {}

    for param_name, param in sig.parameters.items():
        # Skip parameters that should be ignored
        if (
            param_name in ARGS_TO_IGNORE
            or param_name.startswith("_")  # Private/internal params (e.g. ClassVar-backed fields in configs)
            or param.kind == inspect.Parameter.VAR_POSITIONAL
            or param.kind == inspect.Parameter.VAR_KEYWORD
        ):
            continue
        # When a filter is active (e.g. config classes: only own annotations), skip inherited params
        if allowed_params is not None and param_name not in allowed_params:
            continue

        # When a filter is active (e.g. config classes: only own annotations), skip inherited params
        if allowed_params is not None and param_name not in allowed_params:
            continue

        param_name = ARGS_TO_RENAME.get(param_name, param_name)

        # Process parameter type and optional status
        param_type, optional = _process_parameter_type(param)

        # Check for default value
        param_default = ""
        if param.default != inspect._empty and param.default is not None:
            param_default = f", defaults to `{str(param.default)}`"

        param_type, optional_string, shape_string, additional_info, description, is_documented = _get_parameter_info(
            param_name, documented_params, source_args_dict, param_type, optional
        )

        if is_documented:
            if param_name == "config":
                if param_type == "":
                    param_type = f"[`{class_name}`]"
                else:
                    param_type = f"[`{param_type.split('.')[-1]}`]"
            # elif param_type == "" and False:  # TODO: Enforce typing for all parameters
            #     print(f"[ERROR] {param_name} for {func.__qualname__} in file {func.__code__.co_filename} has no type")
            param_type = param_type if "`" in param_type else f"`{param_type}`"
            # Format the parameter docstring
            if additional_info:
                param_docstring = f"{param_name} ({param_type}{additional_info}):{description}"
            else:
                param_docstring = (
                    f"{param_name} ({param_type}{shape_string}{optional_string}{param_default}):{description}"
                )
            docstring += set_min_indent(
                param_docstring,
                indent_level + 8,
            )
        else:
            missing_args[param_name] = {
                "type": param_type if param_type else "<fill_type>",
                "optional": optional,
                "shape": shape_string,
                "description": description if description else "\n    <fill_description>",
                "default": param_default,
            }
            # Try to get the correct source file; for classes decorated with @strict (huggingface_hub),
            # func.__code__.co_filename points to the wrapper in huggingface_hub, not the config file.
            try:
                if parent_class is not None:
                    _source_file = inspect.getsourcefile(parent_class) or func.__code__.co_filename
                else:
                    _source_file = inspect.getsourcefile(inspect.unwrap(func)) or func.__code__.co_filename
            except (TypeError, OSError):
                _source_file = func.__code__.co_filename
            undocumented_parameters.append(
                f"[ERROR] `{param_name}` is part of {func.__qualname__}'s signature, but not documented. Make sure to add it to the docstring of the function in {_source_file}."
            )

    return docstring, missing_args


def find_sig_line(lines, line_end):
    parenthesis_count = 0
    sig_line_end = line_end
    found_sig = False
    while not found_sig:
        for char in lines[sig_line_end]:
            if char == "(":
                parenthesis_count += 1
            elif char == ")":
                parenthesis_count -= 1
                if parenthesis_count == 0:
                    found_sig = True
                    break
        sig_line_end += 1
    return sig_line_end


def _is_image_processor_class(func, parent_class):
    """
    Check if a function belongs to a ProcessorMixin class.

    Uses two methods:
    1. Check parent_class inheritance (if provided)
    2. Check if the source file is named processing_*.py (multimodal processors)
       vs image_processing_*.py, video_processing_*.py, etc. (single-modality processors)

    Args:
        func: The function to check
        parent_class: Optional parent class (if available)

    Returns:
        bool: True if this is a multimodal processor (inherits from ProcessorMixin), False otherwise
    """
    # First, check if parent_class is provided and use it
    if parent_class is not None:
        return "BaseImageProcessor" in parent_class.__name__ or any(
            "BaseImageProcessor" in base.__name__ for base in parent_class.__mro__
        )

    # If parent_class is None, check the filename
    # Multimodal processors are in files named "processing_*.py"
    # Single-modality processors are in "image_processing_*.py", "video_processing_*.py", etc.
    try:
        source_file = inspect.getsourcefile(func)
    except TypeError:
        return False
    if not source_file:
        return False

    # Exception for DummyProcessorForTest
    if func.__qualname__.split(".")[0] == "DummyForTestImageProcessorFast":
        return True

    filename = os.path.basename(source_file)

    # Multimodal processors are implemented in processing_*.py modules
    # (single-modality processors use image_processing_*, video_processing_*, etc.)self.
    return filename.startswith("image_processing_") and filename.endswith(".py")


def _is_processor_class(func, parent_class):
    """
    Check if a function belongs to a ProcessorMixin class.

    Uses two methods:
    1. Check parent_class inheritance (if provided)
    2. Check if the source file is named processing_*.py (multimodal processors)
       vs image_processing_*.py, video_processing_*.py, etc. (single-modality processors)

    Args:
        func: The function to check
        parent_class: Optional parent class (if available)

    Returns:
        bool: True if this is a multimodal processor (inherits from ProcessorMixin), False otherwise
    """
    # First, check if parent_class is provided and use it
    if parent_class is not None:
        return "ProcessorMixin" in parent_class.__name__ or any(
            "ProcessorMixin" in base.__name__ for base in parent_class.__mro__
        )

    # If parent_class is None, check the filename
    # Multimodal processors are in files named "processing_*.py"
    # Single-modality processors are in "image_processing_*.py", "video_processing_*.py", etc.
    try:
        source_file = inspect.getsourcefile(func)
    except TypeError:
        return False
    if not source_file:
        return False

    # Exception for DummyProcessorForTest
    if func.__qualname__.split(".")[0] == "DummyProcessorForTest":
        return True

    filename = os.path.basename(source_file)

    # Multimodal processors are implemented in processing_*.py modules
    # (single-modality processors use image_processing_*, video_processing_*, etc.)self.
    return filename.startswith("processing_") and filename.endswith(".py")


# Python < 3.12 fallback: naming heuristics when __orig_bases__ is not set (cpython#103699).
# Order matters: check ImageProcessorKwargs before ProcessorKwargs.
_BASIC_KWARGS_NAMES = frozenset({"ImagesKwargs", "ProcessingKwargs", "TextKwargs", "VideosKwargs", "AudioKwargs"})
_BASIC_KWARGS_CLASSES = None  # Lazy-loaded name -> class mapping


def _get_base_kwargs_class_from_name(cls_name: str) -> str | None:
    """Map kwargs class name to base using naming conventions. Returns base class name or None."""
    if cls_name in _BASIC_KWARGS_NAMES:
        return cls_name
    if "ImageProcessorKwargs" in cls_name or cls_name.endswith("ImagesKwargs"):
        return "ImagesKwargs"
    if "ProcessorKwargs" in cls_name:
        return "ProcessingKwargs"
    if "VideoProcessorKwargs" in cls_name or cls_name.endswith("VideosKwargs"):
        return "VideosKwargs"
    if "AudioProcessorKwargs" in cls_name or cls_name.endswith("AudioKwargs"):
        return "AudioKwargs"
    if "TextKwargs" in cls_name:
        return "TextKwargs"
    return None


def _get_base_kwargs_class(cls):
    """
    Get the root/base TypedDict class by walking the inheritance chain.
    For model-specific kwargs like ComplexProcessingKwargs(ProcessingKwargs), returns ProcessingKwargs.
    For model-specific kwargs like DummyImageProcessorKwargs(ImagesKwargs), returns ImagesKwargs.

    Compatibility: On Python < 3.12, non-generic TypedDict subclasses do not have __orig_bases__ set
    (cpython#103699). We fall back to naming heuristics (e.g. *ImageProcessorKwargs -> ImagesKwargs).
    """
    current = cls
    while True:
        bases = typing_extensions.get_original_bases(current)
        parent = None
        for base in bases:
            if isinstance(base, type) and base not in (dict, object):
                if getattr(base, "__name__", "") == "TypedDict" and getattr(base, "__module__", "") == "typing":
                    continue
                parent = base
                break
        if parent is None:
            # Python < 3.12 fallback: use naming heuristics
            base_name = _get_base_kwargs_class_from_name(current.__name__)
            if base_name is not None:
                global _BASIC_KWARGS_CLASSES
                if _BASIC_KWARGS_CLASSES is None:
                    from transformers.processing_utils import (
                        AudioKwargs,
                        ImagesKwargs,
                        ProcessingKwargs,
                        TextKwargs,
                        VideosKwargs,
                    )

                    _BASIC_KWARGS_CLASSES = {
                        "ImagesKwargs": ImagesKwargs,
                        "ProcessingKwargs": ProcessingKwargs,
                        "TextKwargs": TextKwargs,
                        "VideosKwargs": VideosKwargs,
                        "AudioKwargs": AudioKwargs,
                    }
                parent = _BASIC_KWARGS_CLASSES[base_name]
        if parent is None or parent == current:
            return current
        current = parent


def _process_kwargs_parameters(sig, func, parent_class, documented_kwargs, indent_level, undocumented_parameters):
    """
    Process **kwargs parameters if needed.

    Args:
        sig (`inspect.Signature`): Function signature
        func (`function`): Function the parameters belong to
        parent_class (`class`): Parent class of the function
        documented_kwargs (`dict`): Dictionary of kwargs that are already documented
        indent_level (`int`): Indentation level
        undocumented_parameters (`list`): List to append undocumented parameters to

    Returns:
        tuple[str, str]: (kwargs docstring, kwargs summary line to add after return_tensors)
    """
    docstring = ""
    kwargs_summary = ""
    # Check if we need to add typed kwargs description to the docstring
    unroll_kwargs = func.__name__ in UNROLL_KWARGS_METHODS
    if not unroll_kwargs and parent_class is not None:
        # Check if the function has a parent class with unroll kwargs
        unroll_kwargs = any(
            any(unroll_kwargs_class in base.__name__ for base in parent_class.__mro__)
            for unroll_kwargs_class in UNROLL_KWARGS_CLASSES
        )
    if not unroll_kwargs:
        return docstring, kwargs_summary

    # Check if this is a processor by inspecting class hierarchy
    is_processor = _is_processor_class(func, parent_class)
    is_image_processor = _is_image_processor_class(func, parent_class)

    # Use appropriate args source based on whether it's a processor or not
    if is_processor:
        source_args_dict = get_args_doc_from_source([ImageProcessorArgs, ProcessorArgs])
    elif is_image_processor:
        source_args_dict = get_args_doc_from_source(ImageProcessorArgs)
    else:
        raise ValueError(
            f"Unrolling kwargs is not supported for {func.__name__} of {parent_class.__name__ if parent_class else 'None'} class"
        )

    # get all unpackable "kwargs" parameters
    kwargs_parameters = [
        kwargs_param
        for _, kwargs_param in sig.parameters.items()
        if kwargs_param.kind == inspect.Parameter.VAR_KEYWORD
    ]
    for kwarg_param in kwargs_parameters:
        # If kwargs not typed, skip
        if kwarg_param.annotation == inspect.Parameter.empty:
            continue

        if not hasattr(kwarg_param.annotation, "__args__") or not hasattr(
            kwarg_param.annotation.__args__[0], "__name__"
        ):
            continue

        if kwarg_param.annotation.__args__[0].__name__ not in BASIC_KWARGS_TYPES:
            # Extract documentation for kwargs
            kwargs_documentation = kwarg_param.annotation.__args__[0].__doc__
            if kwargs_documentation is not None:
                documented_kwargs = parse_docstring(kwargs_documentation)[0]
            # Process each kwarg parameter
            for param_name, param_type_annotation in kwarg_param.annotation.__args__[0].__annotations__.items():
                # Handle nested kwargs structures for processors

                if is_processor and param_name.endswith("_kwargs"):
                    # Check if this is a basic kwargs type that should be skipped
                    # Basic kwargs types are generic containers that shouldn't be documented as individual params

                    # Get the actual type (unwrap Optional if needed)
                    actual_type = param_type_annotation
                    type_name = getattr(param_type_annotation, "__name__", None)
                    if type_name is None and hasattr(param_type_annotation, "__origin__"):
                        # Handle Optional[Type] or Union cases
                        args = getattr(param_type_annotation, "__args__", ())
                        for arg in args:
                            if arg is not type(None):
                                actual_type = arg
                                type_name = getattr(arg, "__name__", None)
                                break

                    # Skip only if it's one of the basic kwargs types
                    if type_name in BASIC_KWARGS_TYPES:
                        continue

                    # Otherwise, unroll the custom typed kwargs
                    # Get the nested TypedDict's annotations
                    if hasattr(actual_type, "__annotations__"):
                        nested_kwargs_doc = getattr(actual_type, "__doc__", None)
                        documented_nested_kwargs = {}
                        if nested_kwargs_doc:
                            documented_nested_kwargs = parse_docstring(nested_kwargs_doc)[0]

                        # Only process fields that are documented in the custom kwargs class's own docstring
                        # This prevents showing too many inherited parameters
                        if not documented_nested_kwargs:
                            # No documentation in the custom kwargs class, skip unrolling
                            continue

                        # Process each field in the custom typed kwargs
                        for nested_param_name, nested_param_type in actual_type.__annotations__.items():
                            # Only document parameters that are explicitly documented in the TypedDict's docstring
                            if nested_param_name not in documented_nested_kwargs:
                                continue
                            nested_param_type_str, nested_optional = process_type_annotation(
                                nested_param_type, nested_param_name
                            )

                            # Check for default value
                            nested_param_default = ""
                            if parent_class is not None:
                                nested_param_default = str(getattr(parent_class, nested_param_name, ""))
                                nested_param_default = (
                                    f", defaults to `{nested_param_default}`" if nested_param_default != "" else ""
                                )

                            # Only use the TypedDict's own docstring, not source_args_dict
                            # This prevents pulling in too many inherited parameters
                            (
                                nested_param_type_str,
                                nested_optional_string,
                                nested_shape_string,
                                nested_additional_info,
                                nested_description,
                                nested_is_documented,
                            ) = _get_parameter_info(
                                nested_param_name,
                                documented_nested_kwargs,
                                {},  # Empty dict - only use TypedDict's own docstring
                                nested_param_type_str,
                                nested_optional,
                            )

                            # nested_is_documented should always be True here since we filter for it above
                            # Check if type is missing
                            if nested_param_type_str == "":
                                print(
                                    f"🚨 {nested_param_name} for {type_name} in file {func.__code__.co_filename} has no type"
                                )
                            nested_param_type_str = (
                                nested_param_type_str if "`" in nested_param_type_str else f"`{nested_param_type_str}`"
                            )
                            # Format the parameter docstring (KWARGS_INDICATOR distinguishes from regular args)
                            if nested_additional_info:
                                docstring += set_min_indent(
                                    f"{nested_param_name} ({nested_param_type_str}{KWARGS_INDICATOR}{nested_additional_info}):{nested_description}",
                                    indent_level + 8,
                                )
                            else:
                                docstring += set_min_indent(
                                    f"{nested_param_name} ({nested_param_type_str}{KWARGS_INDICATOR}{nested_shape_string}{nested_optional_string}{nested_param_default}):{nested_description}",
                                    indent_level + 8,
                                )

                        # Skip processing the _kwargs parameter itself since we've processed its contents
                        continue
                    else:
                        # If we can't get annotations, skip this parameter
                        continue

                if documented_kwargs and param_name not in documented_kwargs:
                    continue
                param_type, optional = process_type_annotation(param_type_annotation, param_name)

                # Check for default value
                param_default = ""
                if parent_class is not None:
                    param_default = str(getattr(parent_class, param_name, ""))
                    param_default = f", defaults to `{param_default}`" if param_default != "" else ""

                param_type, optional_string, shape_string, additional_info, description, is_documented = (
                    _get_parameter_info(param_name, documented_kwargs, source_args_dict, param_type, optional)
                )

                if is_documented:
                    # Check if type is missing
                    if param_type == "":
                        print(
                            f"[ERROR] {param_name} for {kwarg_param.annotation.__args__[0].__qualname__} in file {func.__code__.co_filename} has no type"
                        )
                    param_type = param_type if "`" in param_type else f"`{param_type}`"
                    # Format the parameter docstring (KWARGS_INDICATOR distinguishes from regular args)
                    if additional_info:
                        docstring += set_min_indent(
                            f"{param_name} ({param_type}{KWARGS_INDICATOR}{additional_info}):{description}",
                            indent_level + 8,
                        )
                    else:
                        docstring += set_min_indent(
                            f"{param_name} ({param_type}{KWARGS_INDICATOR}{shape_string}{optional_string}{param_default}):{description}",
                            indent_level + 8,
                        )
                else:
                    undocumented_parameters.append(
                        f"[ERROR] `{param_name}` is part of {kwarg_param.annotation.__args__[0].__qualname__}, but not documented. Make sure to add it to the docstring of the function in {func.__code__.co_filename}."
                    )

        # Build **kwargs summary line (added after return_tensors in _process_parameters_section)
        kwargs_annot_cls = kwarg_param.annotation.__args__[0]
        kwargs_type_name = _get_base_kwargs_class(kwargs_annot_cls).__name__
        kwargs_info = source_args_dict.get("__kwargs__", {})
        kwargs_description = kwargs_info.get(
            "description",
            "Additional keyword arguments. Model-specific parameters are listed above.",
        )
        kwargs_summary = set_min_indent(
            f"**kwargs ([`{kwargs_type_name}`], *optional*):{kwargs_description}",
            indent_level + 8,
        )

    return docstring, kwargs_summary


def _add_return_tensors_to_docstring(func, parent_class, docstring, indent_level):
    """
    Add return_tensors parameter documentation for processor __call__ methods if not already present.

    Args:
        func (`function`): Function being processed
        parent_class (`class`): Parent class of the function
        docstring (`str`): Current docstring being built
        indent_level (`int`): Indentation level

    Returns:
        str: Updated docstring with return_tensors if applicable
    """
    # Check if this is a processor __call__ method or an image processor preprocess method
    is_processor_call = False
    is_image_processor_preprocess = False
    if func.__name__ == "__call__":
        # Check if this is a processor by inspecting class hierarchy
        is_processor_call = _is_processor_class(func, parent_class)

    if func.__name__ == "preprocess":
        is_image_processor_preprocess = _is_image_processor_class(func, parent_class)

    # If it's a processor __call__ method or an image processor preprocess method and return_tensors is not already documented
    if (is_processor_call or is_image_processor_preprocess) and "return_tensors" not in docstring:
        # Get the return_tensors documentation from ImageProcessorArgs
        source_args_dict = (
            get_args_doc_from_source(ProcessorArgs)
            if is_processor_call
            else get_args_doc_from_source(ImageProcessorArgs)
        )
        return_tensors_info = source_args_dict["return_tensors"]
        param_type = return_tensors_info.get("type", "`str` or [`~utils.TensorType`]")
        description = return_tensors_info["description"]

        # Format the parameter type
        param_type = param_type if "`" in param_type else f"`{param_type}`"

        # Format the parameter docstring
        param_docstring = f"return_tensors ({param_type}, *optional*):{description}"
        docstring += set_min_indent(param_docstring, indent_level + 8)

    return docstring


def _process_parameters_section(
    func_documentation,
    sig,
    func,
    class_name,
    model_name_lowercase,
    parent_class,
    indent_level,
    source_args_dict,
    allowed_params,
):
    """
    Process the parameters section of the docstring.

    Args:
        func_documentation (`str`): Existing function documentation (manually specified in the docstring)
        sig (`inspect.Signature`): Function signature
        func (`function`): Function the parameters belong to
        class_name (`str`): Name of the class the function belongs to
        model_name_lowercase (`str`): Lowercase model name
        parent_class (`class`): Parent class of the function (if any)
        indent_level (`int`): Indentation level
    """
    # Start Args section — constant string, min_indent is always 0, so skip set_min_indent
    docstring = " " * (indent_level + 4) + "Args:\n"
    undocumented_parameters = []
    documented_params = {}
    documented_kwargs = {}

    # Parse existing docstring if available
    if func_documentation is not None:
        documented_params, func_documentation = parse_docstring(func_documentation)

    # Process regular parameters
    param_docstring, missing_args = _process_regular_parameters(
        sig,
        func,
        class_name,
        documented_params,
        indent_level,
        undocumented_parameters,
        source_args_dict,
        parent_class,
        allowed_params,
    )
    docstring += param_docstring

    # Process **kwargs parameters if needed
    kwargs_docstring, kwargs_summary = _process_kwargs_parameters(
        sig, func, parent_class, documented_kwargs, indent_level, undocumented_parameters
    )
    docstring += kwargs_docstring

    # Add return_tensors for processor __call__ methods if not already present
    docstring = _add_return_tensors_to_docstring(func, parent_class, docstring, indent_level)

    # Add **kwargs summary line after return_tensors
    docstring += kwargs_summary

    # Report undocumented parameters
    if len(undocumented_parameters) > 0:
        print("\n".join(undocumented_parameters))

    return docstring


def _prepare_return_docstring(output_type, config_class, add_intro=True):
    """
    Prepare the return docstring from a ModelOutput class.

    This is a robust replacement for the old _prepare_output_docstrings from doc.py,
    using the same parsing and formatting methods as the rest of auto_docstring.

    Args:
        output_type: The ModelOutput class to generate documentation for
        config_class (`str`): Config class for the model
        add_intro (`bool`): Whether to add the introduction text

    Returns:
        str: Formatted return docstring
    """
    output_docstring = output_type.__doc__

    # If the class has no docstring, try to use the parent class's docstring
    if output_docstring is None and hasattr(output_type, "__mro__"):
        for base in output_type.__mro__[1:]:  # Skip the class itself
            if base.__doc__ is not None:
                output_docstring = base.__doc__
                break

    if output_docstring is None:
        if add_intro:
            raise ValueError(
                f"No docstring found for `{output_type.__name__}` or its parent classes. "
                "Make sure the ModelOutput class or one of its parents has a docstring."
            )
        return ""

    # Parse the output class docstring to extract parameters
    documented_params, _ = parse_docstring(output_docstring)

    if not documented_params and add_intro:
        raise ValueError(
            f"No `Args` or `Parameters` section is found in the docstring of `{output_type.__name__}`. "
            "Make sure it has a docstring and contains either `Args` or `Parameters`."
        )

    # Build the return section
    full_output_type, _ = process_type_annotation(output_type)
    if add_intro:
        # Import here to avoid circular import
        from .doc import PT_RETURN_INTRODUCTION

        intro = PT_RETURN_INTRODUCTION.format(full_output_type=full_output_type, config_class=config_class)
    else:
        intro = f"Returns:\n    `{full_output_type}`"
        if documented_params:
            intro += ":\n"
        else:
            intro += "\n"

    # Build the parameters section
    params_text = ""
    if documented_params:
        for param_name, param_info in documented_params.items():
            param_type = param_info.get("type", "")
            param_description = param_info.get("description", "").strip()
            additional_info = param_info.get("additional_info", "")

            # Handle types with unbalanced backticks due to nested parentheses
            # The parse_docstring function splits types like `tuple(torch.FloatTensor)` incorrectly
            # so we need to reconstruct the complete type by grabbing the closing part from additional_info
            if param_type.startswith("`") and not param_type.endswith("`"):
                # Find the closing backtick in additional_info
                closing_backtick_idx = additional_info.find("`")
                if closing_backtick_idx != -1:
                    # Grab everything up to and including the closing backtick
                    param_type += additional_info[: closing_backtick_idx + 1]
                    # Remove that part from additional_info
                    additional_info = additional_info[closing_backtick_idx + 1 :]

            # Strip backticks from type to add them back consistently
            param_type = param_type.strip("`")

            # Use process_type_annotation to ensure consistent type formatting
            # This applies the same formatting rules as the rest of auto_docstring
            if param_type:
                param_type, _ = process_type_annotation(param_type)

            # Build the parameter line
            if additional_info:
                # additional_info contains shape and optional status
                param_line = f"- **{param_name}** (`{param_type}`{additional_info}) -- {param_description}"
            else:
                param_line = f"- **{param_name}** (`{param_type}`) -- {param_description}"

            # Handle multi-line descriptions:
            # Split the description to handle continuations with proper indentation
            lines = param_line.split("\n")
            formatted_lines = []
            for i, line in enumerate(lines):
                if i == 0:
                    # First line gets no extra indent (just the bullet point)
                    formatted_lines.append(line)
                else:
                    # Continuation lines: strip existing indentation and add 2 spaces (relative to the bullet)
                    formatted_lines.append("  " + line.lstrip())

            param_text = "\n".join(formatted_lines)

            # Indent everything to 4 spaces and append with newline
            param_text_indented = set_min_indent(param_text, 4)
            params_text += param_text_indented + "\n"

    result = intro + params_text

    return result


def _process_returns_section(func_documentation, sig, config_class, indent_level):
    """
    Process the returns section of the docstring.

    Args:
        func_documentation (`str`): Existing function documentation (manually specified in the docstring)
        sig (`inspect.Signature`): Function signature
        config_class (`str`): Config class for the model
        indent_level (`int`): Indentation level
    """
    return_docstring = ""

    # Extract returns section from existing docstring if available
    if func_documentation is not None and (match_start := _re_return.search(func_documentation)) is not None:
        match_end = _re_example.search(func_documentation)
        if match_end:
            return_docstring = func_documentation[match_start.start() : match_end.start()]
            func_documentation = func_documentation[match_end.start() :]
        else:
            return_docstring = func_documentation[match_start.start() :]
            func_documentation = ""
        return_docstring = set_min_indent(return_docstring, indent_level + 4)
    # Otherwise, generate return docstring from return annotation if available
    elif sig.return_annotation is not None and sig.return_annotation != inspect._empty:
        add_intro, return_annotation = contains_type(sig.return_annotation, ModelOutput)
        return_docstring = _prepare_return_docstring(return_annotation, config_class, add_intro=add_intro)
        # PT_RETURN_INTRODUCTION already starts with \n, so only add blank line if it doesn't start with one
        if not return_docstring.startswith("\n"):
            return_docstring = "\n" + return_docstring
        return_docstring = set_min_indent(return_docstring, indent_level + 4)

    return return_docstring, func_documentation


def _process_example_section(
    func_documentation, func, parent_class, class_name, model_name_lowercase, config_class, checkpoint, indent_level
):
    """
    Process the example section of the docstring.

    Args:
        func_documentation (`str`): Existing function documentation (manually specified in the docstring)
        func (`function`): Function being processed
        parent_class (`class`): Parent class of the function
        class_name (`str`): Name of the class
        model_name_lowercase (`str`): Lowercase model name
        config_class (`str`): Config class for the model
        checkpoint: Checkpoint to use in examples
        indent_level (`int`): Indentation level
    """
    # Import here to avoid circular import
    from transformers.models import auto as auto_module

    example_docstring = ""

    # Use existing example section if available (with or without an "Example:" header)
    if func_documentation is not None and (match := _re_example.search(func_documentation)):
        example_docstring = func_documentation[match.start() :]
        example_docstring = "\n" + set_min_indent(example_docstring, indent_level + 4)
    # Skip examples for processors
    elif _is_processor_class(func, parent_class):
        # Processors don't get auto-generated examples
        return example_docstring
    # No examples for __init__ methods or if the class is not a model
    elif parent_class is None and model_name_lowercase is not None:
        global _re_model_task
        if _re_model_task is None:
            _re_model_task = re.compile(rf"({'|'.join(PT_SAMPLE_DOCSTRINGS.keys())})")
        model_task = _re_model_task.search(class_name)
        CONFIG_MAPPING = auto_module.configuration_auto.CONFIG_MAPPING

        # Get checkpoint example
        if (checkpoint_example := checkpoint) is None:
            try:
                checkpoint_example = get_checkpoint_from_config_class(CONFIG_MAPPING[model_name_lowercase])
            except KeyError:
                # For models with inconsistent lowercase model name
                if model_name_lowercase in HARDCODED_CONFIG_FOR_MODELS:
                    CONFIG_MAPPING_NAMES = auto_module.configuration_auto.CONFIG_MAPPING_NAMES
                    config_class_name = HARDCODED_CONFIG_FOR_MODELS[model_name_lowercase]
                    if config_class_name in CONFIG_MAPPING_NAMES.values():
                        model_name_for_auto_config = [
                            k for k, v in CONFIG_MAPPING_NAMES.items() if v == config_class_name
                        ][0]
                        if model_name_for_auto_config in CONFIG_MAPPING:
                            checkpoint_example = get_checkpoint_from_config_class(
                                CONFIG_MAPPING[model_name_for_auto_config]
                            )

        # Add example based on model task
        if model_task is not None:
            if checkpoint_example is not None:
                example_annotation = ""
                task = model_task.group()
                example_annotation = PT_SAMPLE_DOCSTRINGS[task].format(
                    model_class=class_name,
                    checkpoint=checkpoint_example,
                    expected_output="...",
                    expected_loss="...",
                    qa_target_start_index=14,
                    qa_target_end_index=15,
                    mask="<mask>",
                )
                example_docstring = set_min_indent(example_annotation, indent_level + 4)
            else:
                print(
                    f"[ERROR] No checkpoint found for {class_name}.{func.__name__}. Please add a `checkpoint` arg to `auto_docstring` or add one in {config_class}'s docstring"
                )
        else:
            # Check if the model is in a pipeline to get an example
            for name_model_list_for_task in MODELS_TO_PIPELINE:
                try:
                    model_list_for_task = getattr(auto_module.modeling_auto, name_model_list_for_task)
                except (ImportError, AttributeError):
                    continue
                if class_name in model_list_for_task.values():
                    pipeline_name = MODELS_TO_PIPELINE[name_model_list_for_task]
                    example_annotation = PIPELINE_TASKS_TO_SAMPLE_DOCSTRINGS[pipeline_name].format(
                        model_class=class_name,
                        checkpoint=checkpoint_example,
                        expected_output="...",
                        expected_loss="...",
                        qa_target_start_index=14,
                        qa_target_end_index=15,
                    )
                    example_docstring = set_min_indent(example_annotation, indent_level + 4)
                    break

    return example_docstring


def auto_method_docstring(
    func,
    parent_class=None,
    custom_intro=None,
    custom_args=None,
    checkpoint=None,
    source_args_dict=None,
    allowed_params=None,
):
    """
    Wrapper that automatically generates docstring.
    """

    # Use inspect to retrieve the method's signature
    sig = inspect.signature(func)
    indent_level = get_indent_level(func) if not parent_class else get_indent_level(parent_class)

    # Get model information
    model_name_lowercase, class_name, config_class = _get_model_info(func, parent_class)
    func_documentation = func.__doc__

    if custom_args is not None and func_documentation is not None:
        func_documentation = "\n" + set_min_indent(custom_args.strip("\n"), 0) + "\n" + func_documentation
    elif custom_args is not None:
        func_documentation = "\n" + set_min_indent(custom_args.strip("\n"), 0)

    # Add intro to the docstring before args description if needed
    if custom_intro is not None:
        docstring = set_min_indent(custom_intro, indent_level + 4)
        if not docstring.strip().endswith("\n"):
            docstring += "\n"
    else:
        docstring = add_intro_docstring(func, class_name=class_name, indent_level=indent_level)

    # Process Parameters section
    docstring += _process_parameters_section(
        func_documentation,
        sig,
        func,
        class_name,
        model_name_lowercase,
        parent_class,
        indent_level,
        source_args_dict,
        allowed_params,
    )

    # Process Returns section
    return_docstring, func_documentation = _process_returns_section(
        func_documentation, sig, config_class, indent_level
    )
    docstring += return_docstring

    # Process Example section
    example_docstring = _process_example_section(
        func_documentation,
        func,
        parent_class,
        class_name,
        model_name_lowercase,
        config_class,
        checkpoint,
        indent_level,
    )
    docstring += example_docstring

    # Format the docstring with the placeholders
    docstring = format_args_docstring(docstring, model_name_lowercase)

    # Assign the dynamically generated docstring to the wrapper function
    func.__doc__ = docstring
    return func


def auto_class_docstring(cls, custom_intro=None, custom_args=None, checkpoint=None):
    """
    Wrapper that automatically generates a docstring for classes based on their attributes and methods.
    """
    # import here to avoid circular import
    from transformers.models import auto as auto_module

    is_dataclass = False
    is_processor = False
    is_config = False
    is_image_processor = False
    docstring_init = ""
    docstring_args = ""
    if "PreTrainedModel" in (x.__name__ for x in cls.__mro__):
        docstring_init = auto_method_docstring(
            cls.__init__, parent_class=cls, custom_args=custom_args, checkpoint=checkpoint
        ).__doc__.replace("Args:", "Parameters:")
    elif "ProcessorMixin" in (x.__name__ for x in cls.__mro__):
        is_processor = True
        docstring_init = auto_method_docstring(
            cls.__init__,
            parent_class=cls,
            custom_args=custom_args,
            checkpoint=checkpoint,
            source_args_dict=get_args_doc_from_source([ModelArgs, ImageProcessorArgs, ProcessorArgs]),
        ).__doc__.replace("Args:", "Parameters:")
    elif "ModelOutput" in (x.__name__ for x in cls.__mro__):
        # We have a data class
        is_dataclass = True
        doc_class = cls.__doc__
        if custom_args is None and doc_class:
            custom_args = doc_class
        docstring_args = auto_method_docstring(
            cls.__init__,
            parent_class=cls,
            custom_args=custom_args,
            checkpoint=checkpoint,
            source_args_dict=get_args_doc_from_source(ModelOutputArgs),
        ).__doc__
    elif any("BaseImageProcessor" in x.__name__ for x in cls.__mro__):
        is_image_processor = True
        docstring_init = auto_method_docstring(
            cls.__init__,
            parent_class=cls,
            custom_args=custom_args,
            checkpoint=checkpoint,
            source_args_dict=get_args_doc_from_source(ImageProcessorArgs),
        ).__doc__
    elif "PreTrainedConfig" in (x.__name__ for x in cls.__mro__):
        is_config = True
        doc_class = cls.__doc__
        if custom_args is None and doc_class:
            custom_args = doc_class

        # Collect all non-ClassVar annotations from the class and its ancestors up to
        # (but not including) PreTrainedConfig. This allows inherited params from intermediate
        # config base classes to be documented, while naturally excluding PreTrainedConfig-specific
        # quasi-ClassVar params (e.g. `transformers_version`, `architectures`).
        own_config_params = set()
        for ancestor in cls.__mro__:
            if ancestor.__name__ == "PreTrainedConfig":
                break
            own_config_params |= {
                k for k, v in getattr(ancestor, "__annotations__", {}).items() if get_origin(v) is not ClassVar
            }
        allowed_params = own_config_params if own_config_params else None
        docstring_init = auto_method_docstring(
            cls.__init__,
            parent_class=cls,
            custom_args=custom_args,
            checkpoint=checkpoint,
            source_args_dict=get_args_doc_from_source([ConfigArgs]),
            allowed_params=allowed_params,
        ).__doc__

    indent_level = get_indent_level(cls)
    model_name_lowercase = get_model_name(cls)
    model_name_title = " ".join([k.title() for k in model_name_lowercase.split("_")]) if model_name_lowercase else None
    model_base_class = f"{model_name_title.title()}Model" if model_name_title is not None else None
    if model_name_lowercase is not None:
        try:
            model_base_class = getattr(
                getattr(auto_module, PLACEHOLDER_TO_AUTO_MODULE["model_class"][0]),
                PLACEHOLDER_TO_AUTO_MODULE["model_class"][1],
            )[model_name_lowercase]
        except KeyError:
            pass
        except ImportError:
            # In some environments, certain model classes might not be available. In that case, we can skip this part.
            pass

    if model_name_lowercase and model_name_lowercase not in getattr(
        getattr(auto_module, PLACEHOLDER_TO_AUTO_MODULE["config_class"][0]),
        PLACEHOLDER_TO_AUTO_MODULE["config_class"][1],
    ):
        model_name_lowercase = model_name_lowercase.replace("_", "-")

    name = re.findall(rf"({'|'.join(ClassDocstring.__dict__.keys())})$", cls.__name__)

    if name == [] and custom_intro is None and not is_dataclass and not is_processor and not is_image_processor:
        raise ValueError(
            f"`{cls.__name__}` is not registered in the auto doc. Here are the available classes: {ClassDocstring.__dict__.keys()}.\n"
            "Add a `custom_intro` to the decorator if you want to use `auto_docstring` on a class not registered in the auto doc."
        )
    if name != [] or custom_intro is not None or is_config or is_dataclass or is_processor or is_image_processor:
        name = name[0] if name else None
        formatting_kwargs = {"model_name": model_name_title}
        if name == "Config":
            formatting_kwargs.update({"model_base_class": model_base_class, "model_checkpoint": checkpoint})
        if custom_intro is not None:
            pre_block = equalize_indent(custom_intro, indent_level)
            if not pre_block.endswith("\n"):
                pre_block += "\n"
        elif is_processor:
            # Generate processor intro dynamically
            pre_block = generate_processor_intro(cls)
            if pre_block:
                pre_block = equalize_indent(pre_block, indent_level)
                pre_block = format_args_docstring(pre_block, model_name_lowercase)
        elif is_image_processor:
            pre_block = r"Constructs a {image_processor_class} image processor."
            if pre_block:
                pre_block = equalize_indent(pre_block, indent_level)
                pre_block = format_args_docstring(pre_block, model_name_lowercase)
        elif model_name_title is None or name is None:
            pre_block = ""
        else:
            pre_block = getattr(ClassDocstring, name).format(**formatting_kwargs)
        # Start building the docstring
        docstring = set_min_indent(f"{pre_block}", indent_level) if len(pre_block) else ""
        if name != "PreTrainedModel" and "PreTrainedModel" in (x.__name__ for x in cls.__mro__):
            docstring += set_min_indent(f"{ClassDocstring.PreTrainedModel}", indent_level)
        # Add the __init__ docstring
        if docstring_init:
            docstring += set_min_indent(f"\n{docstring_init}", indent_level)
        elif is_dataclass or is_config:
            # No init function, we have a data class
            docstring += docstring_args if docstring_args else "\nArgs:\n"
            source_args_dict = get_args_doc_from_source(ModelOutputArgs)
            doc_class = cls.__doc__ if cls.__doc__ else ""
            documented_kwargs = parse_docstring(doc_class)[0]
            for param_name, param_type_annotation in cls.__annotations__.items():
                param_type, optional = process_type_annotation(param_type_annotation, param_name)

                # Check for default value
                param_default = ""
                param_default = str(getattr(cls, param_name, ""))
                param_default = f", defaults to `{param_default}`" if param_default != "" else ""

                param_type, optional_string, shape_string, additional_info, description, is_documented = (
                    _get_parameter_info(param_name, documented_kwargs, source_args_dict, param_type, optional)
                )

                if is_documented:
                    # Check if type is missing
                    if param_type == "":
                        print(
                            f"[ERROR] {param_name} for {cls.__qualname__} in file {cls.__code__.co_filename} has no type"
                        )
                    param_type = param_type if "`" in param_type else f"`{param_type}`"
                    # Format the parameter docstring
                    if additional_info:
                        docstring += set_min_indent(
                            f"{param_name} ({param_type}{additional_info}):{description}",
                            indent_level + 8,
                        )
                    else:
                        docstring += set_min_indent(
                            f"{param_name} ({param_type}{shape_string}{optional_string}{param_default}):{description}",
                            indent_level + 8,
                        )
        # TODO (Yoni): Add support for Attributes section in docs

    else:
        print(
            f"You used `@auto_class_docstring` decorator on `{cls.__name__}` but this class is not part of the AutoMappings. Remove the decorator"
        )
    # Assign the dynamically generated docstring to the wrapper class
    cls.__doc__ = docstring

    return cls


def auto_docstring(obj=None, *, custom_intro=None, custom_args=None, checkpoint=None):
    r"""
    Automatically generates comprehensive docstrings for model classes and methods in the Transformers library.

    This decorator reduces boilerplate by automatically including standard argument descriptions while allowing
    overrides to add new or custom arguments. It inspects function signatures, retrieves predefined docstrings
    for common arguments (like `input_ids`, `attention_mask`, etc.), and generates complete documentation
    including examples and return value descriptions.

    For complete documentation and examples, read this [guide](https://huggingface.co/docs/transformers/auto_docstring).

    Examples of usage:

        Basic usage (no parameters):
        ```python
        @auto_docstring
        class MyAwesomeModel(PreTrainedModel):
            def __init__(self, config, custom_parameter: int = 10):
                r'''
                custom_parameter (`int`, *optional*, defaults to 10):
                    Description of the custom parameter for MyAwesomeModel.
                '''
                super().__init__(config)
                self.custom_parameter = custom_parameter
        ```

        Using `custom_intro` with a class:
        ```python
        @auto_docstring(
            custom_intro="This model implements a novel attention mechanism for improved performance."
        )
        class MySpecialModel(PreTrainedModel):
            def __init__(self, config, attention_type: str = "standard"):
                r'''
                attention_type (`str`, *optional*, defaults to "standard"):
                    Type of attention mechanism to use.
                '''
                super().__init__(config)
        ```

        Using `custom_intro` with a method, and specify custom arguments and example directly in the docstring:
        ```python
        @auto_docstring(
            custom_intro="Performs forward pass with enhanced attention computation."
        )
        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            r'''
            custom_parameter (`int`, *optional*, defaults to 10):
                Description of the custom parameter for MyAwesomeModel.

            Example:

            ```python
            >>> model = MyAwesomeModel(config)
            >>> model.forward(input_ids=torch.tensor([1, 2, 3]), attention_mask=torch.tensor([1, 1, 1]))
            ```
            '''
        ```

        Using `custom_args` to define reusable arguments:
        ```python
        VISION_ARGS = r'''
        pixel_values (`torch.FloatTensor`, *optional*):
            Pixel values of the input images.
        image_features (`torch.FloatTensor`, *optional*):
            Pre-computed image features for efficient processing.
        '''

        @auto_docstring(custom_args=VISION_ARGS)
        def encode_images(self, pixel_values=None, image_features=None):
            # ... method implementation
        ```

        Combining `custom_intro` and `custom_args`:
        ```python
        MULTIMODAL_ARGS = r'''
        vision_features (`torch.FloatTensor`, *optional*):
            Pre-extracted vision features from the vision encoder.
        fusion_strategy (`str`, *optional*, defaults to "concat"):
            Strategy for fusing text and vision modalities.
        '''

        @auto_docstring(
            custom_intro="Processes multimodal inputs combining text and vision.",
            custom_args=MULTIMODAL_ARGS
        )
        def forward(
            self,
            input_ids,
            attention_mask=None,
            vision_features=None,
            fusion_strategy="concat"
        ):
            # ... multimodal processing
        ```

        Using with ModelOutput classes:
        ```python
        @dataclass
        @auto_docstring(
            custom_intro="Custom model outputs with additional fields."
        )
        class MyModelOutput(ImageClassifierOutput):
            r'''
            loss (`torch.FloatTensor`, *optional*):
                The loss of the model.
            custom_field (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
                A custom output field specific to this model.
            '''

            # Standard fields like hidden_states, logits, attentions etc. can be automatically documented
            # However, given that the loss docstring is often different per model, you should document it above
            loss: Optional[torch.FloatTensor] = None
            logits: Optional[torch.FloatTensor] = None
            hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
            attentions: Optional[tuple[torch.FloatTensor, ...]] = None
            custom_field: Optional[torch.FloatTensor] = None
        ```

    Args:
        custom_intro (`str`, *optional*):
            Custom introduction text to add to the docstring. This replaces the default
            introduction text generated by the decorator before the Args section. Use this to describe what
            makes your model or method special.
        custom_args (`str`, *optional*):
            Custom argument documentation in docstring format. This allows you to define
            argument descriptions once and reuse them across multiple methods. The format should follow the
            standard docstring convention: `arg_name (`type`, *optional*, defaults to `value`): Description.`
        checkpoint (`str`, *optional*):
            Checkpoint name to use in examples within the docstring. This is typically
            automatically inferred from the model configuration class, but can be overridden if needed for
            custom examples.

    Note:
        - Standard arguments (`input_ids`, `attention_mask`, `pixel_values`, etc.) are automatically documented
          from predefined descriptions and should not be redefined unless their behavior differs in your model.
        - New or custom arguments should be documented in the method's docstring using the `r''' '''` block
          or passed via the `custom_args` parameter.
        - For model classes, the decorator derives parameter descriptions from the `__init__` method's signature
          and docstring.
        - Return value documentation is automatically generated for methods that return ModelOutput subclasses.
    """

    def auto_docstring_decorator(obj):
        if len(obj.__qualname__.split(".")) > 1:
            return auto_method_docstring(
                obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint
            )
        else:
            return auto_class_docstring(obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint)

    if obj:
        return auto_docstring_decorator(obj)

    return auto_docstring_decorator
