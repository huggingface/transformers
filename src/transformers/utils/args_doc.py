import inspect
from functools import wraps

import regex as re


class ConfigArgs:
    vocab_size = r"""
    Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `LlamaModel`.
    """

    hidden_size = r"""
    Dimension of the hidden representations.
    """

    intermediate_size = r"""
    Dimension of the MLP representations.
    """

    num_hidden_layers = r"""
    Number of hidden layers in the Transformer decoder.
    """

    num_attention_heads = r"""
    Number of attention heads for each attention layer in the Transformer decoder.
    """

    num_key_value_heads = r"""
    This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details checkout [this paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `num_attention_heads`.
    """

    hidden_act = r"""
    The non-linear activation function (function or string) in the decoder. Defaults to `"silu"`.
    """

    max_position_embeddings = r"""
    The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens, Llama 2 up to 4096, CodeLlama up to 16384.
    """

    initializer_range = r"""
    The standard deviation of the truncated_normal_initializer for initializing all weight matrices. Defaults to 0.02.
    """

    rms_norm_eps = r"""
    The epsilon used by the rms normalization layers. Defaults to 1e-06.
    """

    use_cache = r"""
    Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`. Defaults to `True`.
    """

    pad_token_id = r"""
    Padding token id.
    """

    bos_token_id = r"""
    Beginning of stream token id. Defaults to 1.
    """

    eos_token_id = r"""
    End of stream token id. Defaults to 2.
    """

    pretraining_tp = r"""
    Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to understand more about it. This value is necessary to ensure exact reproducibility of the pretraining results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232). Defaults to 1.
    """

    tie_word_embeddings = r"""
    Whether to tie weight embeddings. Defaults to `False`.
    """

    rope_theta = r"""
    The base period of the RoPE embeddings. Defaults to 10000.0.
    """

    rope_scaling = r"""
    Dictionary containing the scaling configuration for the RoPE embeddings. If you apply new rope type and expect the model to work on longer `max_position_embeddings`, we recommend updating this value accordingly.

    Expected contents:
    - `rope_type` (`str`): The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope', 'llama3'], with 'default' being the original RoPE implementation.
    - `factor` (`float`, optional): Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In most scaling types, a factor of x will enable the model to handle sequences of length x * original maximum pre-trained length.
    - `original_max_position_embeddings` (`int`, optional): Used with 'dynamic', 'longrope', and 'llama3'. The original max position embeddings used during pretraining.
    - `attention_factor` (`float`, optional): Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention computation. Defaults to the value recommended by the implementation, using the `factor` field to infer the suggested value.
    - `beta_fast` (`float`, optional): Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear ramp function. Defaults to 32.
    - `beta_slow` (`float`, optional): Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear ramp function. Defaults to 1.
    - `short_factor` (`List[float]`, optional): Only used with 'longrope'. The scaling factor to be applied to short contexts (< `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden size divided by the number of attention heads divided by 2.
    - `long_factor` (`List[float]`, optional): Only used with 'longrope'. The scaling factor to be applied to long contexts (< `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden size divided by the number of attention heads divided by 2.
    - `low_freq_factor` (`float`, optional): Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE.
    - `high_freq_factor` (`float`, optional): Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE.
    """

    attention_bias = r"""
    Whether to use a bias in the query, key, value, and output projection layers during self-attention. Defaults to `False`.
    """

    attention_dropout = r"""
    The dropout ratio for the attention probabilities. Defaults to 0.0.
    """







class ModelArgs:
    labels = r"""of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    """

    num_logits_to_keep = r""":
        Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
        `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
        token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
    """

    input_ids = r""":
        Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.
        Indices can be obtained using `AutoTokenizer`. See `PreTrainedTokenizer.encode` and
        `PreTrainedTokenizer.__call__` for details.

        [What are input IDs?](../glossary#input-ids)
    """

    attention_mask = r"""of shape `(batch_size, sequence_length)`, *optional*):
        Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.

        [What are attention masks?](../glossary#attention-mask)

        Indices can be obtained using `AutoTokenizer`. See `PreTrainedTokenizer.encode` and
        `PreTrainedTokenizer.__call__` for details.
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

    hidden_states = r"""): input to the layer of shape `(batch, seq_len, embed_dim)"""

    position_embeddings = r""", *optional*):
        Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
        with `head_dim` being the embedding dimension of each attention head.
    """

    config = r"""):
        Model configuration class with all the parameters of the model. Initializing with a config file does not
        load the weights associated with the model, only the configuration. Check out the
        [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    start_positions = r""" of shape `(batch_size,)`, *optional*):
        Labels for position (index) of the start of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
        are not taken into account for computing the loss.
    """

    end_positions = r""" of shape `(batch_size,)`, *optional*):
        Labels for position (index) of the end of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
        are not taken into account for computing the loss.
    """

    output_router_logits = r"""
        Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
        should not be returned during inference.
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
        The bare {model_camel} Model outputting raw hidden-states without any specific head on top."""

    ForSequenceClassification = r"""
        The {model_name} Model transformer with a sequence classification head on top (linear layer).

        [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
        (e.g. GPT-2) do.

        Since it does classification on the last token, it requires to know the position of the last token. If a
        `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
        no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
        padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
        each row of the batch).
    """

    ForQuestionAnswering = r"""
        The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
        SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """

    ForTokenClassificatio = r"""
        The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
        output) e.g. for Named-Entity-Recognition (NER) tasks.
    """

    Config = r"""
    This is the configuration class to store the configuration of a [`{}Model`] or a [`TF{}Model`]. It is
    used to instantiate a DeBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the {}
    [{}](https://huggingface.co/{}) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

class ClassAttrs:
    base_model_prefix = r"""TODO """
    supports_gradient_checkpointing = r"""TODO """
    _no_split_module = r"""TODO """
    _skip_keys_device_placement = r"""TODO """
    _supports_flash_attn_2 = r"""TODO """
    _supports_sdpa = r"""TODO """
    _supports_cache_class = r"""TODO """
    _supports_quantized_cache = r"""TODO """
    _supports_static_cache = r"""TODO """
    _init_weights = r"""TODO """


ARGS_TO_IGNORE = {"self", "kwargs", "args", "deprecated_arguments"}


def get_indent_level(func):
    # Get the source code of the function
    source_code = inspect.getsource(func)

    # Get the first line of the source (the function definition)
    first_line = source_code.splitlines()[0]

    # Calculate the indentation level (number of spaces at the start)
    indent_level = len(first_line) - len(first_line.lstrip())

    return indent_level


def parse_docstring(docstring):
    args_pattern = re.compile(r'Args:\s*(.*?)\n\n', re.DOTALL)

    args_match = args_pattern.search(docstring)
    args_section = args_match.group(1).strip() if args_match else None

    params = {}
    if args_section:
        param_pattern = re.compile(r'(\w+) (\(.*?\):\s*)(.*?)(?=\n\w|\Z)')
        for param_match in param_pattern.finditer(args_section):
            param_name = param_match.group(1)
            params[param_name] = "".join(param_match.groups()[1:])
    return params


def auto_docstring(func):
    """
    Wrapper that automatically generates docstring using ARG_TO_DOC.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the original function
        return func(*args, **kwargs)

    # Use inspect to retrieve the function's signature
    sig = inspect.signature(func)
    indent_level = get_indent_level(func)
    # Build the docstring dynamically
    docstring = f"{func.__name__}("
    docstring += ", ".join([f"{param}" for param in sig.parameters])
    docstring += ")\n\n"
    docstring += "Args:\n"
    # Adding description for each parameter from ARG_TO_DOC
    undocumented_parameters = []
    documented_params = set()

    func_documentation = func.__doc__
    if func_documentation is not None:
        documented_params = parse_docstring(func_documentation)

    for param_name, param in sig.parameters.items():
        if param_name in ModelArgs.__dict__:
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation
                if "typing" in str(param_type):
                    param_type = str(param_type).split("typing.")[1]
                else:
                    param_type = f"{param_type.__module__}.{param.annotation.__name__}"
            else:
                param_type = ""
            # Check if the parameter has a default value (considered optional)
            # is_optional = param.default != inspect.Parameter.empty

            indented_doc = getattr(ModelArgs, param_name)  # .replace("\n    ", "\n")
            docstring += f"{' '*indent_level}{param_name} (`{param_type}`{indented_doc}\n"
        elif param_name in ARGS_TO_IGNORE:
            continue
        elif param_name in documented_params:
            docstring += f"{' '*indent_level}{param_name} {documented_params[param_name]}\n"
        else:
            undocumented_parameters.append(
                f"🚨 `{param_name}` is part of {func.__qualname__}'s signature, but not documented. Make sure to add it to the docstring of the function in {func.__code__.co_filename}."
            )

    if len(undocumented_parameters)>0:
        print("\n".join(undocumented_parameters))
    if func.__doc__ is not None:
        docstring += func.__doc__
    # Assign the dynamically generated docstring to the wrapper function
    wrapper.__doc__ = docstring
    return wrapper


def auto_class_docstring(cls):
    """
    Wrapper that automatically generates a docstring for classes based on their attributes and methods.
    """

    class Wrapper(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    indent_level = get_indent_level(cls) + 8

    name = re.findall(rf"({'|'.join(ClassDocstring.__dict__.keys())})", cls.__name__)[0]
    pre_block = getattr(ClassDocstring, name)
    # Start building the docstring
    docstring = f"{pre_block}\n\n"
    attr_docs = ""
    # Get all attributes and methods of the class
    for attr_name, attr_value in cls.__dict__.items():
        if not callable(attr_value) and not attr_name.startswith("__"):
            if attr_value.__class__.__name__ == "property":
                attr_type = "property"
            else:
                attr_type = type(attr_value).__name__
            if "Config" in name:
                attribute_mapping = ConfigArgs
            else:
                attribute_mapping = ClassAttrs
            indented_doc = getattr(attribute_mapping, attr_name, "")
            attr_docs += f"{' ' * (indent_level+4)}{attr_name} (`{attr_type}`): {indented_doc}\n"
    if len(attr_docs.replace(" ", "")):
        docstring += f"{' ' * indent_level}Attributes:\n" + attr_docs
    # Assign the dynamically generated docstring to the wrapper class
    if cls.__doc__ is not None:
        docstring += cls.__doc__
    Wrapper.__doc__ = docstring

    return Wrapper
