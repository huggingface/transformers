

ARGS_TO_DOC = {
    "labels": r""""of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.""",
    "num_logits_to_keep":r"""
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.""",
    "input_ids": r"""
    Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.
    Indices can be obtained using `AutoTokenizer`. See `PreTrainedTokenizer.encode` and
    `PreTrainedTokenizer.__call__` for details.

    [What are input IDs?](../glossary#input-ids)
    """,
    "attention_mask": r"""of shape `(batch_size, sequence_length)`, *optional*):
    Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

    - 1 for tokens that are **not masked**,
    - 0 for tokens that are **masked**.

    [What are attention masks?](../glossary#attention-mask)

    Indices can be obtained using `AutoTokenizer`. See `PreTrainedTokenizer.encode` and
    `PreTrainedTokenizer.__call__` for details.
    """,
    "position_ids": r"""
    Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

    [What are position IDs?](../glossary#position-ids)
    """,
    "past_key_values": r"""
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
    """,
    "past_key_value":r"""deprecated in favor of `past_key_values`""",
    "inputs_embeds": r"""
    Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
    is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
    model's internal embedding lookup matrix.
    """,
    "use_cache": r"""
    If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
    `past_key_values`).
    """,
    "output_attentions": r"""
    Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
    tensors for more detail.
    """,
    "output_hidden_states": r"""
    Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
    more detail.
    """,
    "return_dict": r"""
    Whether or not to return a `~utils.ModelOutput` instead of a plain tuple.
    """,
    "cache_position": r"""
    Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
    this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
    the complete sequence length.
    """,
    "hidden_states": r"""): input to the layer of shape `(batch, seq_len, embed_dim)""", #TODO how do you handle != naming for special models -> merge doc
    "position_embeddings": r""", *optional*):
    Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
    with `head_dim` being the embedding dimension of each attention head.""",
    "config": r""")
    Model configuration class with all the parameters of the model. Initializing with a config file does not
    load the weights associated with the model, only the configuration. Check out the
    [`~PreTrainedModel.from_pretrained`] method to load the model weights.""",
    "start_positions": r""" of shape `(batch_size,)`, *optional*):
    Labels for position (index) of the start of the labelled span for computing the token classification loss.
    Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
    are not taken into account for computing the loss.""",
    "end_positions": r""" of shape `(batch_size,)`, *optional*):
    Labels for position (index) of the end of the labelled span for computing the token classification loss.
    Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
    are not taken into account for computing the loss.""",
}

ARGS_TO_IGNORE = {
    "self", "kwargs", "args"
}

COMMON_FOR_QA_DOCSTRING = """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
""",

COMMON_SEQ_DOC = """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """

COMMON_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""