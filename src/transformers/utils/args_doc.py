# coding=utf-8
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

import inspect
import os
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple, Union, get_args

import regex as re

from .doc import (
    MODELS_TO_PIPELINE,
    PIPELINE_TASKS_TO_SAMPLE_DOCSTRINGS,
    PT_SAMPLE_DOCSTRINGS,
    _prepare_output_docstrings,
)
from .generic import ModelOutput


PATH_TO_TRANSFORMERS = Path("src").resolve() / "transformers"


AUTODOC_FILES = [
    "configuration_*.py",
    "modeling_*.py",
    "tokenization_*.py",
    "processing_*.py",
    "image_processing_*_fast.py",
    "image_processing_*.py",
    "feature_extractor_*.py",
]

PLACEHOLDER_TO_AUTO_MODULE = {
    "image_processor_class": ("image_processing_auto", "IMAGE_PROCESSOR_MAPPING_NAMES"),
    "feature_extractor_class": ("feature_extraction_auto", "FEATURE_EXTRACTOR_MAPPING_NAMES"),
    "processor_class": ("processing_auto", "PROCESSOR_MAPPING_NAMES"),
    "config_class": ("configuration_auto", "CONFIG_MAPPING_NAMES"),
}

UNROLL_KWARGS_METHODS = {
    "preprocess",
}

UNROLL_KWARGS_CLASSES = {
    "ImageProcessorFast",
}

HARDCODED_CONFIG_FOR_MODELS = {
    "openai": "OpenAIGPTConfig",
    "x-clip": "XCLIPConfig",
    "kosmos2": "Kosmos2Config",
    "donut": "DonutSwinConfig",
    "esmfold": "EsmConfig",
}

_re_checkpoint = re.compile(r"\[(.+?)\]\((https://huggingface\.co/.+?)\)")


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
    Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
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
    into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
    soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and
    conversion into a tensor of type `torch.FloatTensor`. See [`{processor_class}.__call__`] for details.
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

    head_mask = {
        "description": """
    Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

    - 1 indicates the head is **not masked**,
    - 0 indicates the head is **masked**.
    """,
        "shape": "of shape `(num_heads,)` or `(num_layers, num_heads)`",
    }

    cross_attn_head_mask = {
        "description": """
    Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

    - 1 indicates the head is **not masked**,
    - 0 indicates the head is **masked**.
    """,
        "shape": "of shape `(num_layers, num_heads)`",
    }

    decoder_attention_mask = {
        "description": """
    Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
    make sure the model can only look at previous inputs in order to predict the future.
    """,
        "shape": "of shape `(batch_size, target_sequence_length)`",
    }

    decoder_head_mask = {
        "description": """
    Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

    - 1 indicates the head is **not masked**,
    - 0 indicates the head is **masked**.
    """,
        "shape": "of shape `(decoder_layers, decoder_attention_heads)`",
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

    Two formats are allowed:
        - a [`~cache_utils.Cache`] instance, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
        - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
        shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
        cache format.

    The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
    legacy cache format will be returned.

    If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
    have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
    of shape `(batch_size, sequence_length)`.
    """,
        "shape": None,
    }

    past_key_value = {
        "description": """
    deprecated in favor of `past_key_values`
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

    cache_position = {
        "description": """
    Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
    this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
    the complete sequence length.
    """,
        "shape": "of shape `(sequence_length)`",
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


class ClassDocstring:
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

    ImageProcessorFast = r"""
    Constructs a fast {model_name} image processor.
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
    _supports_flash_attn_2 = r"""
    Whether the model's attention implementation supports FlashAttention 2.0.
    """
    _supports_sdpa = r"""
    Whether the model's attention implementation supports SDPA (Scaled Dot Product Attention).
    """
    _supports_flex_attn = r"""
    Whether the model's attention implementation supports FlexAttention.
    """
    _supports_cache_class = r"""
    Whether the model supports a `Cache` instance as `past_key_values`.
    """
    _supports_quantized_cache = r"""
    Whether the model supports a `QuantoQuantizedCache` instance as `past_key_values`.
    """
    _supports_static_cache = r"""
    Whether the model supports a `StaticCache` instance as `past_key_values`.
    """
    _supports_attention_backend = r"""
    Whether the model supports attention interface functions. This flag signal that the model can be used as an efficient backend in TGI and vLLM.
    """
    _tied_weights_keys = r"""
    A list of `state_dict` keys that are potentially tied to another key in the state_dict.
    """
    # fmt: on


ARGS_TO_IGNORE = {"self", "kwargs", "args", "deprecated_arguments"}


def get_indent_level(func):
    # Use this instead of `inspect.getsource(func)` as getsource can be very slow
    return (len(func.__qualname__.split(".")) - 1) * 4


def equalize_indent(docstring, indent_level):
    """
    Adjust the indentation of a docstring to match the specified indent level.
    """
    # fully dedent the docstring
    docstring = "\n".join([line.lstrip() for line in docstring.splitlines()])
    return textwrap.indent(docstring, " " * indent_level)


def set_min_indent(docstring, indent_level):
    """
    Adjust the indentation of a docstring to match the specified indent level.
    """
    return textwrap.indent(textwrap.dedent(docstring), " " * indent_level)


def parse_shape(docstring):
    shape_pattern = re.compile(r"(of shape\s*(?:`.*?`|\(.*?\)))")
    match = shape_pattern.search(docstring)
    if match:
        return " " + match.group(1)
    return None


def parse_default(docstring):
    default_pattern = re.compile(r"(defaults to \s*[^)]*)")
    match = default_pattern.search(docstring)
    if match:
        return " " + match.group(1)
    return None


def parse_docstring(docstring, max_indent_level=0):
    """
    Parse the docstring to extract the Args section and return it as a dictionary.
    The docstring is expected to be in the format:
    Args:
        arg1 (type): Description of arg1.
        arg2 (type): Description of arg2.

    # This function will also return the remaining part of the docstring after the Args section.
    Returns:/Example:
    ...
    """
    match = re.search(r"(?m)^([ \t]*)(?=Example|Return)", docstring)
    if match:
        remainder_docstring = docstring[match.start() :]
        docstring = docstring[: match.start()]
    else:
        remainder_docstring = ""
    args_pattern = re.compile(r"(?:Args:)(\n.*)?(\n)?$", re.DOTALL)

    args_match = args_pattern.search(docstring)
    # still try to find args description in the docstring, if args are not preceded by "Args:"
    args_section = args_match.group(1).lstrip("\n") if args_match else docstring
    if args_section.split("\n")[-1].strip() == '"""':
        args_section = "\n".join(args_section.split("\n")[:-1])
    if args_section.split("\n")[0].strip() == 'r"""' or args_section.split("\n")[0].strip() == '"""':
        args_section = "\n".join(args_section.split("\n")[1:])
    args_section = set_min_indent(args_section, 0)

    params = {}
    if args_section:
        param_pattern = re.compile(
            # |--- Group 1 ---|| Group 2 ||- Group 3 -||---------- Group 4 ----------|
            rf"^\s{{0,{max_indent_level}}}(\w+)\s*\(\s*([^, \)]*)(\s*.*?)\s*\)\s*:\s*((?:(?!\n^\s{{0,{max_indent_level}}}\w+\s*\().)*)",
            re.DOTALL | re.MULTILINE,
        )
        for match in param_pattern.finditer(args_section):
            param_name = match.group(1)
            param_type = match.group(2)
            # param_type = match.group(2).replace("`", "")
            additional_info = match.group(3)
            optional = "optional" in additional_info
            shape = parse_shape(additional_info)
            default = parse_default(additional_info)
            param_description = match.group(4).strip()
            # set first line of param_description to 4 spaces:
            param_description = re.sub(r"^", " " * 4, param_description, 1)
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

    return params, remainder_docstring


def contains_type(type_hint, target_type) -> Tuple[bool, Optional[object]]:
    """
    Check if a "nested" type hint contains a specific target type,
    return the first-level type containing the target_type if found.
    """
    args = get_args(type_hint)
    if args == ():
        try:
            return issubclass(type_hint, target_type), type_hint
        except Exception as _:
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
    if path.split(os.path.sep)[-3] != "models":
        return None
    file_name = path.split(os.path.sep)[-1]
    for file_type in AUTODOC_FILES:
        start = file_type.split("*")[0]
        end = file_type.split("*")[-1] if "*" in file_type else ""
        if file_name.startswith(start) and file_name.endswith(end):
            model_name_lowercase = file_name[len(start) : -len(end)]
            return model_name_lowercase
    else:
        print(f"🚨 Something went wrong trying to find the model name in the path: {path}")
        return "model"


def get_placeholders_dict(placeholders: List, model_name: str) -> dict:
    """
    Get the dictionary of placeholders for the given model name.
    """
    # import here to avoid circular import
    from transformers.models import auto as auto_module

    placeholders_dict = {}
    for placeholder in placeholders:
        # Infer placeholders from the model name and the auto modules
        if placeholder in PLACEHOLDER_TO_AUTO_MODULE:
            place_holder_value = getattr(
                getattr(auto_module, PLACEHOLDER_TO_AUTO_MODULE[placeholder][0]),
                PLACEHOLDER_TO_AUTO_MODULE[placeholder][1],
            )[model_name]
            if isinstance(place_holder_value, (list, tuple)):
                place_holder_value = place_holder_value[0]
            placeholders_dict[placeholder] = place_holder_value

    return placeholders_dict


def format_args_docstring(args, model_name):
    """
    Replaces placeholders such as {image_processor_class} in the docstring with the actual values,
    deducted from the model name and the auto modules.
    """
    # first check if there are any placeholders in the args, if not return them as is
    placeholders = set(re.findall(r"{(.*?)}", "".join((args[arg]["description"] for arg in args))))
    if not placeholders:
        return args

    # get the placeholders dictionary for the given model name
    placeholders_dict = get_placeholders_dict(placeholders, model_name)

    # replace the placeholders in the args with the values from the placeholders_dict
    for arg in args:
        new_arg = args[arg]["description"]
        placeholders = re.findall(r"{(.*?)}", new_arg)
        placeholders = [placeholder for placeholder in placeholders if placeholder in placeholders_dict]
        if placeholders:
            new_arg = new_arg.format(**{placeholder: placeholders_dict[placeholder] for placeholder in placeholders})
        args[arg]["description"] = new_arg

    return args


def source_args_doc(args_classes: Union[object, List[object]]) -> dict:
    if isinstance(args_classes, (list, tuple)):
        args_classes_dict = {}
        for args_class in args_classes:
            args_classes_dict.update(args_class.__dict__)
        return args_classes_dict
    return args_classes.__dict__


def get_checkpoint_from_config_class(config_class):
    checkpoint = None

    # source code of `config_class`
    # config_source = inspect.getsource(config_class)
    config_source = config_class.__doc__
    checkpoints = _re_checkpoint.findall(config_source)
    # Each `checkpoint` is a tuple of a checkpoint name and a checkpoint link.
    # For example, `('google-bert/bert-base-uncased', 'https://huggingface.co/google-bert/bert-base-uncased')`
    for ckpt_name, ckpt_link in checkpoints:
        # allow the link to end with `/`
        if ckpt_link.endswith("/"):
            ckpt_link = ckpt_link[:-1]

        # verify the checkpoint name corresponds to the checkpoint link
        ckpt_link_from_name = f"https://huggingface.co/{ckpt_name}"
        if ckpt_link == ckpt_link_from_name:
            checkpoint = ckpt_name
            break

    return checkpoint


def add_intro_docstring(func, class_name, parent_class=None, indent_level=0):
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
                    f"🚨 Config not found for {model_name_lowercase}. You can manually add it to HARDCODED_CONFIG_FOR_MODELS in utils/args_doc.py"
                )

    return model_name_lowercase, class_name, config_class


def _process_parameter_type(param, param_name, func):
    """
    Process and format a parameter's type annotation.

    Args:
        param (`inspect.Parameter`): The parameter from the function signature
        param_name (`str`): The name of the parameter
        func (`function`): The function the parameter belongs to
    """
    optional = False
    if param.annotation != inspect.Parameter.empty:
        param_type = param.annotation
        if "typing" in str(param_type):
            param_type = "".join(str(param_type).split("typing.")).replace("transformers.", "~")
        elif hasattr(param_type, "__module__"):
            param_type = f"{param_type.__module__.replace('transformers.', '~').replace('builtins', '')}.{param.annotation.__name__}"
            if param_type[0] == ".":
                param_type = param_type[1:]
        else:
            if False:
                print(
                    f"🚨 {param_type} for {param_name} of {func.__qualname__} in file {func.__code__.co_filename} has an invalid type"
                )
        if "ForwardRef" in param_type:
            param_type = re.sub(r"ForwardRef\('([\w.]+)'\)", r"\1", param_type)
        if "Optional" in param_type:
            param_type = re.sub(r"Optional\[(.*?)\]", r"\1", param_type)
            optional = True
    else:
        param_type = ""

    return param_type, optional


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

    if param_name in documented_params:
        # Parameter is documented in the function's docstring
        if param_type == "" and documented_params[param_name].get("type", None) is not None:
            param_type = documented_params[param_name]["type"]
        optional = documented_params[param_name]["optional"]
        shape = documented_params[param_name]["shape"]
        shape_string = shape if shape else ""
        additional_info = documented_params[param_name]["additional_info"] or ""
        description = f"{documented_params[param_name]['description']}\n"
    elif param_name in source_args_dict:
        # Parameter is documented in ModelArgs or ImageProcessorArgs
        shape = source_args_dict[param_name]["shape"]
        shape_string = " " + shape if shape else ""
        description = source_args_dict[param_name]["description"]
        additional_info = None
    else:
        # Parameter is not documented
        is_documented = False
    optional_string = r", *optional*" if optional else ""

    return param_type, optional_string, shape_string, additional_info, description, is_documented


def _process_regular_parameters(sig, func, class_name, documented_params, indent_level, undocumented_parameters):
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
    source_args_dict = source_args_doc([ModelArgs, ImageProcessorArgs])
    missing_args = {}

    for param_name, param in sig.parameters.items():
        # Skip parameters that should be ignored
        if (
            param_name in ARGS_TO_IGNORE
            or param.kind == inspect.Parameter.VAR_POSITIONAL
            or param.kind == inspect.Parameter.VAR_KEYWORD
        ):
            continue

        # Process parameter type and optional status
        param_type, optional = _process_parameter_type(param, param_name, func)

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
            elif param_type == "" and False:  # TODO: Enforce typing for all parameters
                print(f"🚨 {param_name} for {func.__qualname__} in file {func.__code__.co_filename} has no type")
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
            undocumented_parameters.append(
                f"🚨 `{param_name}` is part of {func.__qualname__}'s signature, but not documented. Make sure to add it to the docstring of the function in {func.__code__.co_filename}."
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


def _process_kwargs_parameters(
    sig, func, parent_class, model_name_lowercase, documented_kwargs, indent_level, undocumented_parameters
):
    """
    Process **kwargs parameters if needed.

    Args:
        sig (`inspect.Signature`): Function signature
        func (`function`): Function the parameters belong to
        parent_class (`class`): Parent class of the function
        model_name_lowercase (`str`): Lowercase model name
        documented_kwargs (`dict`): Dictionary of kwargs that are already documented
        indent_level (`int`): Indentation level
        undocumented_parameters (`list`): List to append undocumented parameters to
    """
    docstring = ""
    source_args_dict = source_args_doc(ImageProcessorArgs)

    # Check if we need to add typed kwargs description to the docstring
    unroll_kwargs = func.__name__ in UNROLL_KWARGS_METHODS
    if not unroll_kwargs and parent_class is not None:
        # Check if the function has a parent class with unroll kwargs
        unroll_kwargs = any(
            unroll_kwargs_class in parent_class.__name__ for unroll_kwargs_class in UNROLL_KWARGS_CLASSES
        )

    if unroll_kwargs:
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

            # Extract documentation for kwargs
            kwargs_documentation = kwarg_param.annotation.__args__[0].__doc__
            if kwargs_documentation is not None:
                documented_kwargs, _ = parse_docstring(kwargs_documentation)
                if model_name_lowercase is not None:
                    documented_kwargs = format_args_docstring(documented_kwargs, model_name_lowercase)

            # Process each kwarg parameter
            for param_name, param_type_annotation in kwarg_param.annotation.__args__[0].__annotations__.items():
                param_type = str(param_type_annotation)
                optional = False

                # Process parameter type
                if "typing" in param_type:
                    param_type = "".join(param_type.split("typing.")).replace("transformers.", "~")
                else:
                    param_type = f"{param_type.replace('transformers.', '~').replace('builtins', '')}.{param_name}"
                if "ForwardRef" in param_type:
                    param_type = re.sub(r"ForwardRef\('([\w.]+)'\)", r"\1", param_type)
                if "Optional" in param_type:
                    param_type = re.sub(r"Optional\[(.*?)\]", r"\1", param_type)
                    optional = True

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
                            f"🚨 {param_name} for {kwarg_param.annotation.__args__[0].__qualname__} in file {func.__code__.co_filename} has no type"
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
                else:
                    undocumented_parameters.append(
                        f"🚨 `{param_name}` is part of {kwarg_param.annotation.__args__[0].__qualname__}, but not documented. Make sure to add it to the docstring of the function in {func.__code__.co_filename}."
                    )

    return docstring


def _process_parameters_section(
    func_documentation, sig, func, class_name, model_name_lowercase, parent_class, indent_level
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
    # Start Args section
    docstring = set_min_indent("Args:\n", indent_level + 4)
    undocumented_parameters = []
    documented_params = {}
    documented_kwargs = {}

    # Parse existing docstring if available
    if func_documentation is not None:
        documented_params, func_documentation = parse_docstring(func_documentation)
        if model_name_lowercase is not None:
            documented_params = format_args_docstring(documented_params, model_name_lowercase)

    # Process regular parameters
    param_docstring, missing_args = _process_regular_parameters(
        sig, func, class_name, documented_params, indent_level, undocumented_parameters
    )
    docstring += param_docstring

    # Process **kwargs parameters if needed
    kwargs_docstring = _process_kwargs_parameters(
        sig, func, parent_class, model_name_lowercase, documented_kwargs, indent_level, undocumented_parameters
    )
    docstring += kwargs_docstring

    # Report undocumented parameters
    if len(undocumented_parameters) > 0:
        print("\n".join(undocumented_parameters))

    return docstring


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
    if (
        func_documentation is not None
        and (match_start := re.search(r"(?m)^([ \t]*)(?=Return)", func_documentation)) is not None
    ):
        match_end = re.search(r"(?m)^([ \t]*)(?=Example)", func_documentation)
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
        return_docstring = _prepare_output_docstrings(return_annotation, config_class, add_intro=add_intro)
        return_docstring = return_docstring.replace("typing.", "")
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

    # Use existing example section if available

    if func_documentation is not None and (match := re.search(r"(?m)^([ \t]*)(?=Example)", func_documentation)):
        example_docstring = func_documentation[match.start() :]
        example_docstring = "\n" + set_min_indent(example_docstring, indent_level + 4)
    # No examples for __init__ methods or if the class is not a model
    elif parent_class is None and model_name_lowercase is not None:
        task = rf"({'|'.join(PT_SAMPLE_DOCSTRINGS.keys())})"
        model_task = re.search(task, class_name)
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
                    f"🚨 No checkpoint found for {class_name}.{func.__name__}. Please add a `checkpoint` arg to `auto_docstring` or add one in {config_class}'s docstring"
                )
        else:
            # Check if the model is in a pipeline to get an example
            for name_model_list_for_task in MODELS_TO_PIPELINE:
                model_list_for_task = getattr(auto_module.modeling_auto, name_model_list_for_task)
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


def auto_method_docstring(func, parent_class=None, custom_intro=None, custom_args=None, checkpoint=None):
    """
    Wrapper that automatically generates docstring.
    """

    # Use inspect to retrieve the method's signature
    sig = inspect.signature(func)
    indent_level = get_indent_level(func)

    # Get model information
    model_name_lowercase, class_name, config_class = _get_model_info(func, parent_class)
    func_documentation = func.__doc__
    if custom_args is not None and func_documentation is not None:
        func_documentation = set_min_indent(custom_args, indent_level + 4) + "\n" + func_documentation
    elif custom_args is not None:
        func_documentation = custom_args

    # Add intro to the docstring before args description if needed
    if custom_intro is not None:
        docstring = set_min_indent(custom_intro, indent_level + 4)
    else:
        docstring = add_intro_docstring(
            func, class_name=class_name, parent_class=parent_class, indent_level=indent_level
        )

    # Process Parameters section
    docstring += _process_parameters_section(
        func_documentation, sig, func, class_name, model_name_lowercase, parent_class, indent_level
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

    # Assign the dynamically generated docstring to the wrapper function
    func.__doc__ = docstring
    return func


def auto_class_docstring(cls, custom_intro=None, custom_args=None, checkpoint=None):
    """
    Wrapper that automatically generates a docstring for classes based on their attributes and methods.
    """
    # import here to avoid circular import
    from transformers.models import auto as auto_module

    docstring_init = auto_method_docstring(cls.__init__, parent_class=cls, custom_args=custom_args).__doc__.replace(
        "Args:", "Parameters:"
    )
    indent_level = get_indent_level(cls)
    model_name_lowercase = get_model_name(cls)
    model_name_title = " ".join([k.title() for k in model_name_lowercase.split("_")]) if model_name_lowercase else None
    if model_name_lowercase and model_name_lowercase not in getattr(
        getattr(auto_module, PLACEHOLDER_TO_AUTO_MODULE["config_class"][0]),
        PLACEHOLDER_TO_AUTO_MODULE["config_class"][1],
    ):
        model_name_lowercase = model_name_lowercase.replace("_", "-")

    name = re.findall(rf"({'|'.join(ClassDocstring.__dict__.keys())})$", cls.__name__)
    if name == [] and cls.__doc__ is None and custom_intro is None:
        raise ValueError(
            f"`{cls.__name__}` is not part of the auto doc. Here are the available classes: {ClassDocstring.__dict__.keys()}"
        )
    if name != [] or custom_intro is not None:
        name = name[0] if name else None
        if custom_intro is not None:
            pre_block = equalize_indent(custom_intro, indent_level)
            if not pre_block.endswith("\n"):
                pre_block += "\n"
        elif model_name_title is None:
            pre_block = ""
        else:
            pre_block = getattr(ClassDocstring, name).format(model_name=model_name_title)
        # Start building the docstring
        docstring = set_min_indent(f"{pre_block}", indent_level) if len(pre_block) else ""
        if name != "PreTrainedModel" and "PreTrainedModel" in (x.__name__ for x in cls.__mro__):
            docstring += set_min_indent(f"{ClassDocstring.PreTrainedModel}", indent_level)
        # Add the __init__ docstring
        docstring += set_min_indent(f"\n{docstring_init}", indent_level)
        attr_docs = ""
        # Get all attributes and methods of the class

        for attr_name, attr_value in cls.__dict__.items():
            if not callable(attr_value) and not attr_name.startswith("__"):
                if attr_value.__class__.__name__ == "property":
                    attr_type = "property"
                else:
                    attr_type = type(attr_value).__name__
                if name and "Config" in name:
                    raise ValueError("Config should have explicit docstring")
                indented_doc = getattr(ClassAttrs, attr_name, None)
                if indented_doc is not None:
                    attr_docs += set_min_indent(f"{attr_name} (`{attr_type}`): {indented_doc}", 0)

        # TODO: Add support for Attributes section in docs
        # if len(attr_docs.replace(" ", "")):
        #     docstring += set_min_indent("\nAttributes:\n", indent_level)
        #     docstring += set_min_indent(attr_docs, indent_level + 4)
    else:
        print(
            f"You used `@auto_class_docstring` decorator on `{cls.__name__}` but this class is not part of the AutoMappings. Remove the decorator"
        )
    # Assign the dynamically generated docstring to the wrapper class
    cls.__doc__ = docstring

    return cls


def auto_docstring(obj=None, *, custom_intro=None, custom_args=None, checkpoint=None):
    """
    Automatically generates docstrings for classes and methods in the Transformers library.

    This decorator can be used in the following forms:
    @auto_docstring
    def my_function(...):
        ...
    or
    @auto_docstring()
    def my_function(...):
        ...
    or
    @auto_docstring(custom_intro="Custom intro", ...)
    def my_function(...):
        ...

    Args:
        custom_intro (str, optional): Custom introduction text to add to the docstring. This will replace the default
            introduction text generated by the decorator before the Args section.
        checkpoint (str, optional): Checkpoint name to use in the docstring. This should be automatically inferred from the
            model configuration class, but can be overridden if needed.
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
