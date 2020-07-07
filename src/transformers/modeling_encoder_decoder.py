# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Classes to support Encoder-Decoder architectures """


import logging
from typing import Optional

from .configuration_encoder_decoder import EncoderDecoderConfig
from .configuration_utils import PretrainedConfig
from .modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)


class EncoderDecoderModel(PreTrainedModel):
    r"""
        :class:`~transformers.EncoderDecoder` is a generic model class that will be
        instantiated as a transformer architecture with one of the base model
        classes of the library as encoder and another one as
        decoder when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method for the encoder and `AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)` class method for the decoder.
    """
    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        assert config is not None or (
            encoder is not None and decoder is not None
        ), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
                config, self.config_class
            )
        # initialize with config
        super().__init__(config)

        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            from transformers import AutoModelForCausalLM

            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder
        assert (
            self.encoder.get_output_embeddings() is None
        ), "The encoder {} should not have a LM Head. Please use a model without LM Head"

    def tie_weights(self):
        # for now no weights tying in encoder-decoder
        pass

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r""" Instantiates an encoder and a decoder from one or two base classes of the library from pre-trained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).
        To train the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                information necessary to initiate the encoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/encoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                information necessary to initiate the decoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/decoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            kwargs: (`optional`) Remaining dictionary of keyword arguments.
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

        Examples::

            >>> from transformers import EncoderDecoderModel
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert
        """

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            assert (
                encoder_pretrained_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"
            from .modeling_auto import AutoModel

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)
        encoder.config.is_decoder = False

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            assert (
                decoder_pretrained_model_name_or_path is not None
            ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"
            from .modeling_auto import AutoModelForCausalLM

            if "config" not in kwargs_decoder:
                from transformers import AutoConfig

                decoder_config = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder_config.is_decoder is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attribute `is_decoder` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` is set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        return cls(encoder=encoder, decoder=decoder)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        head_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        **kwargs,
    ):

        """
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the encoder.
                Indices can be obtained using :class:`transformers.PretrainedTokenizer`.
                See :func:`transformers.PreTrainedTokenizer.encode` and
                :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Mask to avoid performing attention on padding token indices for the encoder.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            head_mask: (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
                Mask to nullify selected heads of the self-attention modules for the encoder.
                Mask values selected in ``[0, 1]``:
                ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
            encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
                Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
                `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
                Used in the cross-attention of the decoder.
            decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
                Provide for sequence to sequence training to the decoder.
                Indices can be obtained using :class:`transformers.PretrainedTokenizer`.
                See :func:`transformers.PreTrainedTokenizer.encode` and
                :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
            decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
                Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
            decoder_head_mask: (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
                Mask to nullify selected heads of the self-attention modules for the decoder.
                Mask values selected in ``[0, 1]``:
                ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
            decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
                Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the masked language modeling loss for the decoder.
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                in ``[0, ..., config.vocab_size]``
            kwargs: (`optional`) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:
                - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.
                - With a `decoder_` prefix which will be input as `**decoder_kwargs` for the decoder forward function.

        Examples::

            >>> from transformers import EncoderDecoderModel, BertTokenizer
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert

            >>> # forward
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

            >>> # training
            >>> loss, outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)[:2]

            >>> # generation
            >>> generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)

        """

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                **kwargs_encoder,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            labels=labels,
            **kwargs_decoder,
        )

        return decoder_outputs + encoder_outputs

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if type(past) is tuple:
            encoder_outputs, _ = past
        else:
            encoder_outputs = (past,)

        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)

        return {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
        }

    def _reorder_cache(self, past, beam_idx):
        # as a default encoder-decoder models do not re-order the past.
        # TODO(PVP): might have to be updated, e.g. if GPT2 is to be used as a decoder
        return past
