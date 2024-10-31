# coding=utf-8
# Copyright 2024 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from transformers import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.utils import auto_class_docstring, auto_docstring


LLAMA_CLM_FORWARD = """The [`LlamaForCausalLM`] forward method, overrides the `__call__` special method.

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


        Returns:
            [`transformers.modeling_outputs.CausalLMOutputWithPast`] or `tuple(torch.FloatTensor)`: A [`transformers.modeling_outputs.CausalLMOutputWithPast`] or a tuple of
            `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
            elements depending on the configuration ([`LlamaConfig`]) and inputs.

            - **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
            - **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            - **past_key_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

                Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
                `past_key_values` input) to speed up sequential decoding.
            - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (onefor the output of the embeddings, if the model has an embedding layer, +
                one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
                sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.


        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```
"""

LLAMA_DECODER = """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model"""


class AutoDocstringTest(unittest.TestCase):
    def test_modeling_docstring(self):
        llama_docstring = (
            "\n        The bare Llama Model outputting raw hidden-states without any specific head on top.\n\n"
        )
        self.assertEqual(llama_docstring, LlamaModel.__doc__)

        llama_docstring = """"Args:\n    input_ids (`torch.LongTensor`):\n        Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.\n        Indices can be obtained using `AutoTokenizer`. See `PreTrainedTokenizer.encode` and\n        `PreTrainedTokenizer.__call__` for details.\n\n        [What are input IDs?](../glossary#input-ids)\n    \n    attention_mask (`Optional[torch.Tensor]`) of shape `(batch_size, sequence_length)`:\n        Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n        - 1 for tokens that are **not masked**,\n        - 0 for tokens that are **masked**.\n\n        [What are attention masks?](../glossary#attention-mask)\n\n        Indices can be obtained using `AutoTokenizer`. See `PreTrainedTokenizer.encode` and\n        `PreTrainedTokenizer.__call__` for details.\n    \n    position_ids (`Optional[torch.LongTensor]`):\n        Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.\n\n        [What are position IDs?](../glossary#position-ids)\n    \n    past_key_values (`Union[~cache_utils.Cache, List[torch.FloatTensor], NoneType]`):\n        Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention\n        blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`\n        returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.\n\n        Two formats are allowed:\n            - a `~cache_utils.Cache` instance, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);\n            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of\n            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy\n            cache format.\n\n        The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the\n        legacy cache format will be returned.\n\n        If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't\n        have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`\n        of shape `(batch_size, sequence_length)`.\n    \n    inputs_embeds (`Optional[torch.FloatTensor]`):\n        Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n        is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n        model's internal embedding lookup matrix.\n    \n    use_cache (`Optional[bool]`):\n        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n        `past_key_values`).\n    \n    output_attentions (`Optional[bool]`):\n        Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n        tensors for more detail.\n    \n    output_hidden_states (`Optional[bool]`):\n        Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n        more detail.\n    \n    return_dict (`Optional[bool]`):\n        Whether or not to return a `~utils.ModelOutput` instead of a plain tuple.\n    \n    cache_position (`Optional[torch.LongTensor]`):\n        Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,\n        this tensor is not affected by padding. It is used to update the cache in the correct position and to infer\n        the complete sequence length.\n    \n"""
        self.assertEqual(llama_docstring, LlamaModel.forward.__doc__)
        self.assertEqual(LLAMA_CLM_FORWARD, LlamaForCausalLM.forward.__doc__)
        self.assertEqual(LLAMA_DECODER, LlamaDecoderLayer.forward.__doc__)

    def test_auto_doc(self):
        COOL_CLASS_DOC = """
        Args:
            input_ids (some):
            flash_attn_kwargs (FlashAttentionKwrargs):
                parameters that are completely optional and that should be passed.
            another_warg (something): should pass
            and_another_on (this time):
                I want
                this to be
                quite long

        Example

        ```python
        >>> import
        ```
        """

        @auto_class_docstring
        class MyCoolClass:
            @auto_docstring
            def __init__(input_ids, flash_attn_kwargs=None, another_warg=True, and_another_on=1):
                r"""
                Args:
                    flash_attn_kwargs (FlashAttentionKwrargs):
                        parameters that are completely optional and that should be passed.
                    another_warg (something): should pass
                    and_another_on (this time):
                        I want
                        this to be
                        quite long

                Example

                ```python
                >>> import
                ```
                """
                pass

        self.assertEqual(MyCoolClass.__init__.__doc__, COOL_CLASS_DOC)
