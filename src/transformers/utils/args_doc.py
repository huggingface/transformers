import inspect
import os
import textwrap
from functools import wraps
from typing import Optional, Tuple, get_args

import regex as re

from .doc import PT_SAMPLE_DOCSTRINGS, _prepare_output_docstrings
from .generic import ModelOutput


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
    "processor_class": ("processing_auto", "PROCESSOR_MAPPING_NAMES"),
    "config_class": ("configuration_auto", "CONFIG_MAPPING_NAMES"),
}

UNROLL_KWARGS_METHODS = {
    "preprocess",
}

UNROLL_KWARGS_CLASSES = {
    "ImageProcessorFast",
}


class ImageProcessorArgs:
    images = r""":
    Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
    passing in images with pixel values between 0 and 1, set `do_rescale=False`.
    """

    do_resize = r""":
    Whether to resize the image.
    """

    size = r""":
    Describes the maximum input dimensions to the model.
    """

    default_to_square = r""":
    Whether to default to a square image when resizing, if size is an int.
    """

    resample = r""":
    Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
    has an effect if `do_resize` is set to `True`.
    """

    do_center_crop = r""":
    Whether to center crop the image.
    """

    crop_size = r""":
    Size of the output image after applying `center_crop`.
    """

    do_rescale = r""":
    Whether to rescale the image.
    """

    rescale_factor = r""":
    Rescale factor to rescale the image by if `do_rescale` is set to `True`.
    """

    do_normalize = r""":
    Whether to normalize the image.
    """

    image_mean = r""":
    Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
    """

    image_std = r""":
    Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
    `True`.
    """

    do_convert_rgb = r""":
    Whether to convert the image to RGB.
    """

    return_tensors = r""":
    Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
    """

    data_format = r""":
    Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
    """

    input_data_format = r""":
    The channel dimension format for the input image. If unset, the channel dimension format is inferred
    from the input image. Can be one of:
    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
    - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
    """

    device = r""":
    The device to process the images on. If unset, the device is inferred from the input images.
    """


class ModelArgs:
    labels = r""" of shape `(batch_size, sequence_length)`:
    Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
    config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
    (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    """

    num_logits_to_keep = r""":
    Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
    `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
    token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
    """

    input_ids = r"""of shape `(batch_size, sequence_length)`):
    Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

    Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
    [`PreTrainedTokenizer.__call__`] for details.

    [What are input IDs?](../glossary#input-ids)
    """

    attention_mask = r""" of shape `(batch_size, sequence_length)`:
    Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

    - 1 for tokens that are **not masked**,
    - 0 for tokens that are **masked**.

    [What are attention masks?](../glossary#attention-mask)
    """

    token_type_ids = r""" of shape `(batch_size, input_ids_length)`:
    Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
    1]`:

    - 0 corresponds to a *sentence A* token,
    - 1 corresponds to a *sentence B* token.

    [What are token type IDs?](../glossary#token-type-ids)
    """

    position_ids = r""":
    Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

    [What are position IDs?](../glossary#position-ids)
    """

    past_key_values = r""":
    Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
    blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
    returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

    Two formats are allowed:
        - a `~cache_utils.Cache` instance, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
        - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
        shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
        cache format.

    The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
    legacy cache format will be returned.

    If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
    have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
    of shape `(batch_size, sequence_length)`.
    """

    past_key_value = r""":deprecated in favor of `past_key_values`"""

    inputs_embeds = r""":
    Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
    is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
    model's internal embedding lookup matrix.
    """

    use_cache = r""":
    If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
    `past_key_values`).
    """

    output_attentions = r""":
    Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
    tensors for more detail.
    """

    output_hidden_states = r""":
    Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
    more detail.
    """

    return_dict = r""":
    Whether or not to return a `~utils.ModelOutput` instead of a plain tuple.
    """

    cache_position = r""":
    Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
    this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
    the complete sequence length.
    """

    hidden_states = r""": input to the layer of shape `(batch, seq_len, embed_dim)"""

    position_embeddings = r""":
    Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
    with `head_dim` being the embedding dimension of each attention head.
    """

    config = r""":
    Model configuration class with all the parameters of the model. Initializing with a config file does not
    load the weights associated with the model, only the configuration. Check out the
    [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    start_positions = r""" of shape `(batch_size,)`:
    Labels for position (index) of the start of the labelled span for computing the token classification loss.
    Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
    are not taken into account for computing the loss.
    """

    end_positions = r""" of shape `(batch_size,)`:
    Labels for position (index) of the end of the labelled span for computing the token classification loss.
    Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
    are not taken into account for computing the loss.
    """

    output_router_logits = r"""
    Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
    should not be returned during inference.
    """

    pixel_values = r""" of shape `(batch_size, num_channels, image_size, image_size)):
    The tensors corresponding to the input images. Pixel values can be obtained using
    [`{image_processor_class}`]. See [`{image_processor_class}.__call__`] for details ([`{processor_class}`] uses
    [`{image_processor_class}`] for processing images).
    """


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

    Decoder = r"""
    The bare {model_name} Decoder outputting raw hidden-states without any specific head on top.
    """

    TextModel = r"""
    The bare {model_name} Text Model outputting raw hidden-states without any specific head on to.
    """

    ForSequenceClassification = r"""
    The {model_name} Model with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """

    ForQuestionAnswering = r"""
    The {model_name} transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
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

    Config = r"""
    This is the configuration class to store the configuration of a [`{model_name}Model`] or a [`TF{model_name}Model`]. It is
    used to instantiate a DeBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the {model_name}
    [{}](https://huggingface.co/{model_checkpoint}) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
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
    # Get the source code of the function
    source_code = inspect.getsource(func)

    # Get the first line of the source (the function definition)
    first_line = source_code.splitlines()[0]

    # Calculate the indentation level (number of spaces at the start)
    indent_level = len(first_line) - len(first_line.lstrip())

    return indent_level


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


def parse_docstring(docstring):
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
    args_pattern = re.compile(r"(Args:)(\n.*)?(\n)?$", re.DOTALL)

    args_match = args_pattern.search(docstring)
    args_section = args_match.group(2).lstrip("\n") if args_match else None

    params = {}
    if args_section:
        param_pattern = re.compile(
            r"^\s*(\w+)\s*\((.*?)\):\s*(.*?)(?=\n^\s*\w+\s*\(|\n\s*$)", re.DOTALL | re.MULTILINE
        )
        for match in param_pattern.finditer(args_section):
            param_name = match.group(1)
            param_type = match.group(2)
            param_description = match.group(3).strip()
            param_description = equalize_indent(f"\n{param_description}\n", 4)
            params[param_name] = {"type": param_type, "description": param_description}
    docstring, subs_made = re.subn(r"Args:[\S\s]*(?=Example|Return)", "", docstring)
    if not subs_made:
        docstring = re.sub(r"Args:[\S\s]*", "", docstring)
    return params, docstring


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


def format_args_docstring(args, model_name):
    """
    Replaces placeholders such as {image_processor_class} in the docstring with the actual values,
    deducted from the model name and the auto modules.
    """
    # import here to avoid circular import
    from transformers.models import auto as auto_module

    # first check if there are any placeholders in the args, if not return them as is
    placeholders = set(re.findall(r"{(.*?)}", "".join((args[arg]["description"] for arg in args))))
    if not placeholders:
        return args

    # Infer placeholders from the model name and the auto modules
    placeholders_dict = {}
    for placeholder in placeholders:
        if placeholder in PLACEHOLDER_TO_AUTO_MODULE:
            place_holder_value = getattr(
                getattr(auto_module, PLACEHOLDER_TO_AUTO_MODULE[placeholder][0]),
                PLACEHOLDER_TO_AUTO_MODULE[placeholder][1],
            )[model_name]
            if isinstance(place_holder_value, (list, tuple)):
                place_holder_value = place_holder_value[0]
            placeholders_dict[placeholder] = place_holder_value

    # replace the placeholders in the args with the values from the placeholders_dict
    for arg in args:
        new_arg = args[arg]["description"]
        placeholders = re.findall(r"{(.*?)}", new_arg)
        if placeholders:
            new_arg = new_arg.format(**{placeholder: placeholders_dict[placeholder] for placeholder in placeholders})
        args[arg]["description"] = new_arg

    return args


def auto_docstring(obj):
    if len(obj.__qualname__.split(".")) > 1:
        return auto_method_docstring(obj)
    else:
        return auto_class_docstring(obj)


def auto_method_docstring(func, parent_class=None):
    """
    Wrapper that automatically generates docstring using ARG_TO_DOC.
    """
    # import here to avoid circular import
    from transformers.models import auto as auto_module

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Use inspect to retrieve the function's signature
    sig = inspect.signature(func)
    indent_level = get_indent_level(func)
    if parent_class is not None:
        model_name_lowercase = get_model_name(parent_class)
    else:
        model_name_lowercase = get_model_name(func)
    class_name = func.__qualname__.split(".")[0]
    config_class = getattr(
        getattr(auto_module, PLACEHOLDER_TO_AUTO_MODULE["config_class"][0]),
        PLACEHOLDER_TO_AUTO_MODULE["config_class"][1],
    )[model_name_lowercase]

    docstring = ""
    if func.__name__ == "forward":
        docstring_forward = rf"""The [`{class_name}`] forward method, overrides the `__call__` special method.

        <Tip>

        Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
        instance afterwards instead of this since the former takes care of running the pre and post processing steps while
        the latter silently ignores them.

        </Tip>

        """

        docstring += equalize_indent(docstring_forward, indent_level + 4)

    # Build the docstring dynamically
    docstring += set_min_indent("Args:\n", indent_level + 4)
    # Adding description for each parameter from ARG_TO_DOC
    undocumented_parameters = []
    documented_params = {}
    documented_kwargs = {}

    func_documentation = func.__doc__

    # ------ Args section ------

    if func_documentation is not None:
        documented_params, func_documentation = parse_docstring(func_documentation)
        documented_params = format_args_docstring(documented_params, model_name_lowercase)

    for param_name, param in sig.parameters.items():
        if (
            param_name in ARGS_TO_IGNORE
            or param.kind == inspect.Parameter.VAR_POSITIONAL
            or param.kind == inspect.Parameter.VAR_KEYWORD
        ):
            continue
        if param.annotation != inspect.Parameter.empty:
            param_type = param.annotation
            if "typing" in str(param_type):
                param_type = "".join(str(param_type).split("typing.")).replace("transformers.", "~")
            else:
                param_type = f"{param_type.__module__.replace('transformers.','~').replace('builtins','')}.{param.annotation.__name__}"
            if "ForwardRef" in param_type:
                param_type = re.sub(r"ForwardRef\('([\w.]+)'\)", r"\1", param_type)
        else:
            param_type = ""

        # Check if the parameter has a default value (considered optional)
        param_default = ""
        if param.default != inspect._empty and param.default is not None:
            param_default = f", defaults to `{str(param.default)}`"

        if param_name in documented_params:
            if param_type == "" and documented_params[param_name].get("type", None) is not None:
                param_type = documented_params[param_name]["type"]
            docstring += set_min_indent(
                f"{param_name} (`{param_type}`{param_default}):{documented_params[param_name]['description']}\n",
                indent_level + 8,
            )
        elif param_name in ModelArgs.__dict__:
            indented_doc = getattr(ModelArgs, param_name)
            docstring += set_min_indent(
                f"{param_name} (`{param_type}`{param_default}){indented_doc}", indent_level + 8
            )
        elif param_name in ImageProcessorArgs.__dict__:
            indented_doc = getattr(ImageProcessorArgs, param_name)
            docstring += set_min_indent(
                f"{param_name} (`{param_type}`{param_default}){indented_doc}", indent_level + 8
            )
        else:
            undocumented_parameters.append(
                f"🚨 `{param_name}` is part of {func.__qualname__}'s signature, but not documented. Make sure to add it to the docstring of the function in {func.__code__.co_filename}."
            )

    # Check if we need to add typed kwargs description to the docstring
    unroll_kwargs = func.__name__ in UNROLL_KWARGS_METHODS
    if not unroll_kwargs:
        # Check if the function has a parent class with unroll kwargs
        if parent_class is not None:
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
            kwargs_documentation = kwarg_param.annotation.__args__[0].__doc__
            documented_kwargs, _ = parse_docstring(kwargs_documentation)
            documented_kwargs = format_args_docstring(documented_kwargs, model_name_lowercase)
            for param_name, param_type in kwarg_param.annotation.__args__[0].__annotations__.items():
                param_type = str(param_type)
                if "typing" in param_type:
                    param_type = "".join(param_type.split("typing.")).replace("transformers.", "~")
                else:
                    param_type = f"{param_type.__module__.replace('transformers.','~').replace('builtins','').replace()}.{param_name}"
                if "ForwardRef" in param_type:
                    param_type = re.sub(r"ForwardRef\('([\w.]+)'\)", r"\1", param_type)
                # Check if the parameter has a default value (considered optional)
                # is_optional = param.default != inspect.Parameter.empty
                param_default = ""
                if parent_class is not None:
                    param_default = str(getattr(parent_class, param_name, ""))
                    param_default = f", defaults to `{param_default}`" if param_default != "" else ""

                if param_name in documented_kwargs:
                    if param_type == "" and documented_kwargs[param_name].get("type", None) is not None:
                        param_type = documented_kwargs[param_name]["type"]
                    docstring += set_min_indent(
                        f"{param_name} (`{param_type}`{param_default}):{documented_kwargs[param_name]['description']}\n",
                        indent_level + 8,
                    )
                elif param_name in ImageProcessorArgs.__dict__:
                    indented_doc = getattr(ImageProcessorArgs, param_name)
                    docstring += set_min_indent(
                        f"{param_name} (`{param_type}`{param_default}){indented_doc}", indent_level + 8
                    )
                else:
                    undocumented_parameters.append(
                        f"🚨 `{param_name}` is part of {func.__qualname__}'s signature, but not documented. Make sure to add it to the docstring of the function in {func.__code__.co_filename}."
                    )

    # ------ Returns section ------

    if sig.return_annotation is not None and sig.return_annotation != inspect._empty:
        add_intro, return_annotation = contains_type(sig.return_annotation, ModelOutput)
        return_docstring = _prepare_output_docstrings(return_annotation, config_class, add_intro=add_intro)
        docstring += set_min_indent(return_docstring, indent_level + 4)

    # ------ Example section ------

    task = rf"({'|'.join(PT_SAMPLE_DOCSTRINGS.keys())})"
    model_task = re.search(task, class_name)
    example_annotation = ""
    if model_task is not None:
        task = model_task.group()
        example_annotation = PT_SAMPLE_DOCSTRINGS[task]

    docstring += example_annotation

    if len(undocumented_parameters) > 0:
        print("\n".join(undocumented_parameters))
    if func_documentation is not None:
        docstring += func_documentation

    # Assign the dynamically generated docstring to the wrapper function
    wrapper.__doc__ = docstring
    return wrapper


def auto_class_docstring(cls):
    """
    Wrapper that automatically generates a docstring for classes based on their attributes and methods.
    """
    docstring_init = auto_method_docstring(cls.__init__, parent_class=cls).__doc__.replace("Args:", "Parameters:")
    indent_level = get_indent_level(cls)
    model_name_lowercase = get_model_name(cls)
    model_name_title = "".join([k.title() for k in model_name_lowercase.split("_")])

    name = re.findall(rf"({'|'.join(ClassDocstring.__dict__.keys())})", cls.__name__)
    if name == [] and cls.__doc__ is None:
        raise ValueError(
            f"`{cls.__name__}` is not part of the auto doc. Here are the available classes: {ClassDocstring.__dict__.keys()}"
        )
    if name != []:
        name = name[0]
        pre_block = getattr(ClassDocstring, name).format(model_name=model_name_title, model_checkpoint="dummy-path")
        # Start building the docstring
        docstring = set_min_indent(f"{pre_block}", indent_level)
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
                if "Config" in name:
                    raise ValueError("Config should have explicit docstring")
                indented_doc = getattr(ClassAttrs, attr_name, None)
                if indented_doc is not None:
                    attr_docs += set_min_indent(f"{attr_name} (`{attr_type}`): {indented_doc}", 0)
        if len(attr_docs.replace(" ", "")):
            docstring += set_min_indent("\nAttributes:\n", indent_level)
            docstring += set_min_indent(attr_docs, indent_level + 4)
    else:
        print(
            f"You used `@auto_class_docstring` decorator on `{cls.__name__}` but this class is not part of the AutoMappings. Remove the decorator"
        )
    # Assign the dynamically generated docstring to the wrapper class
    if cls.__doc__ is not None:
        docstring += cls.__doc__
    cls.__doc__ = docstring

    return cls
