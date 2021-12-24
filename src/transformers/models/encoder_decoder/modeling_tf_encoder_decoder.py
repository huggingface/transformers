# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
""" Classes to support TF Encoder-Decoder architectures """


import tempfile
from typing import Optional

import tensorflow as tf

from ...configuration_utils import PretrainedConfig
from ...file_utils import (
    DUMMY_INPUTS,
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqLMOutput
from ...modeling_tf_utils import TFPreTrainedModel, get_initializer, input_processing
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_tf_auto import TFAutoModel, TFAutoModelForCausalLM
from .configuration_encoder_decoder import EncoderDecoderConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "EncoderDecoderConfig"

ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the
    encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via
    [`~TFAutoModel.from_pretrained`] function and the decoder is loaded via
    [`~TFAutoModelForCausalLM.from_pretrained`] function. Cross-attention layers are automatically
    added to the decoder and should be fine-tuned on a downstream generative task, like summarization.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models
    (see the examples for more information).

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`EncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the
            model weights.
"""

ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`np.ndarray` or `tf.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            Provide for sequence to sequence training to the decoder. Indices can be obtained using
            [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
        decoder_attention_mask (`np.ndarray` or `tf.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will
            also be used by default.
        encoder_outputs (`tuple(tuple(tf.Tensor)`, *optional*):
            This tuple must consist of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
            `attentions`) `last_hidden_state` (`tf.Tensor` of shape `({0}, hidden_size)`) is a
            tensor of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the
            decoder.
        past_key_values (`tuple(tuple(tf.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape `(batch_size, 1)`
            instead of all `decoder_input_ids` of shape `({0})`.
        inputs_embeds (`np.ndarray` or `tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (`np.ndarray` or `tf.Tensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert `decoder_input_ids`
            indices into associated vectors than the model's internal embedding lookup matrix.
        labels (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Labels for computing the masked language modeling loss for the decoder. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up
            decoding (see `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~file_utils.Seq2SeqLMOutput`] instead of a
            plain tuple.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
        kwargs: (*optional*) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:

            - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.
            - With a *decoder_* prefix which will be input as `**decoder_kwargs`` for the decoder forward function.
"""


@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
class TFEncoderDecoderModel(TFPreTrainedModel):
    r"""
    [`TFEncoderDecoder`] is a generic model class that will be instantiated as a transformer
    architecture with one of the base model classes of the library as encoder and another one as decoder when created
    with the :meth*~transformers.TFAutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.TFAutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"
    load_weight_prefix = "tf_encoder_decoder_model_1"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[TFPreTrainedModel] = None,
        decoder: Optional[TFPreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, "
                    "it has to be equal to the encoder's `hidden_size`. "
                    f"Got {config.decoder.cross_attention_hidden_size} for `config.decoder.cross_attention_hidden_size` "
                    f"and {config.encoder.hidden_size} for `config.encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if encoder is None:
            encoder = TFAutoModel.from_config(config.encoder, name="encoder")

        if decoder is None:
            decoder = TFAutoModelForCausalLM.from_config(config.decoder, name="decoder")

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config: {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config: {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # encoder outputs might need to be projected to different dimension for decoder
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = tf.keras.layers.Dense(
                units=self.decoder.config.hidden_size,
                kernel_initializer=get_initializer(config.encoder.initializer_range),
                name="enc_to_dec_proj",
            )

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        # Add `decoder_input_ids` because `self.decoder` requires it.
        input_ids = tf.constant(DUMMY_INPUTS)
        dummy = {"input_ids": input_ids, "decoder_input_ids": input_ids}
        # Add `encoder_hidden_states` to make the cross-attention layers' weights initialized
        if self.config.add_cross_attention:
            batch_size, seq_len = input_ids.shape
            shape = (batch_size, seq_len) + (self.config.hidden_size,)
            h = tf.random.uniform(shape=shape)
            dummy["encoder_hidden_states"] = h

        return dummy

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Initializing *TFEncoderDecoderModel* from a pytorch checkpoint is not supported currently.

        If there are only pytorch checkpoints for a particular encoder-decoder model, a workaround is:

        ```python
        >>> # a workaround to load from pytorch checkpoint
        >>> _model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")
        >>> _model.encoder.save_pretrained("./encoder")
        >>> _model.decoder.save_pretrained("./decoder")
        >>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
        ... )
        >>> # This is only for copying some specific attributes of this particular model.
        >>> model.config = _model.config
        ```

        Example:

        ```python
        >>> from transformers import TFEncoderDecoderModel
        >>> model = TFEncoderDecoderModel.from_pretrained("ydshieh/bert2bert-cnn_dailymail-fp16")
        ```"""

        from_pt = kwargs.pop("from_pt", False)
        if from_pt:
            raise ValueError(
                "Initializing `TFEncoderDecoderModel` from a pytorch checkpoint is not supported currently. "
                "Use a tensorflow checkpoint instead. If only the pytorch checkpoints are available, "
                "create the encoder and decoder models separately, and use them to initialize `TFEncoderDecoderModel`. "
                "Check `TFEncoderDecoderModel.from_encoder_decoder_pretrained()` for more details."
            )

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> TFPreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        Params:
            encoder_pretrained_model_name_or_path (:obj: *str*, *optional*):
                Information necessary to initiate the encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under
                      a user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~TFPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *pytorch index checkpoint file* (e.g, `./pt_model/`). In this case,
                      `encoder_from_pt` should be set to `True`.

            decoder_pretrained_model_name_or_path (:obj: *str*, *optional*, defaults to *None*):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under
                      a user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~TFPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *pytorch checkpoint file* (e.g, `./pt_model/`). In this case,
                      `decoder_from_pt` should be set to `True`.

            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import TFEncoderDecoderModel
        >>> # initialize a bert2gpt2 from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
        >>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'gpt2')
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./bert2gpt2")
        >>> # load fine-tuned model
        >>> model = TFEncoderDecoderModel.from_pretrained("./bert2gpt2")
        ```"""

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config = AutoConfig.from_pretrained(encoder_pretrained_model_name_or_path)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            kwargs_encoder["name"] = "encoder"
            kwargs_encoder["load_weight_prefix"] = cls.load_weight_prefix
            encoder = TFAutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

            # This is necessary to make `from_pretrained` following `save_pretrained` work correctly
            if kwargs_encoder.get("from_pt", None):
                del kwargs_encoder["from_pt"]
                with tempfile.TemporaryDirectory() as tmp_dirname:
                    encoder.save_pretrained(tmp_dirname)
                    del encoder
                    encoder = TFAutoModel.from_pretrained(tmp_dirname, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. "
                        f"Cross attention layers are added to {decoder_pretrained_model_name_or_path} "
                        f"and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for "
                        "cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            kwargs_decoder["name"] = "decoder"
            kwargs_decoder["load_weight_prefix"] = cls.load_weight_prefix
            decoder = TFAutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

            # This is necessary to make `from_pretrained` following `save_pretrained` work correctly
            if kwargs_decoder.get("from_pt", None):
                del kwargs_decoder["from_pt"]
                with tempfile.TemporaryDirectory() as tmp_dirname:
                    decoder.save_pretrained(tmp_dirname)
                    del decoder
                    decoder = TFAutoModelForCausalLM.from_pretrained(tmp_dirname, **kwargs_decoder)

        # Make sure these 2 `tf.keras.Model` have fixed names so `from_pretrained` could load model weights correctly.
        if encoder.name != "encoder":
            raise ValueError("encoder model must be created with the name `encoder`.")
        if decoder.name != "decoder":
            raise ValueError("decoder model must be created with the name `decoder`.")

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config)

    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import TFEncoderDecoderModel, BertTokenizer

        >>> # initialize a bert2gpt2 from a pretrained BERT and GPT2 models. Note that the cross-attention layers will be randomly initialized
        >>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-cased', 'gpt2')

        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        >>> # forward
        >>> input_ids = tokenizer.encode("Hello, my dog is cute", add_special_tokens=True, return_tensors='tf')  # Batch size 1
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

        >>> # training
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
        >>> loss, logits = outputs.loss, outputs.logits

        >>> # save and load from pretrained
        >>> model.save_pretrained("bert2gpt2")
        >>> model = TFEncoderDecoderModel.from_pretrained("bert2gpt2")

        >>> # generation
        >>> generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.bos_token_id)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # Let the user be responsible for the expected format.
        if encoder_outputs is not None:
            if return_dict and not isinstance(encoder_outputs, ModelOutput):
                raise ValueError(
                    "If `return_dict=True` and `encoder_outputs` is provided, it should be an instance of "
                    f"`ModelOutput`. Got an instance {type(encoder_outputs)} for `encoder_outputs`."
                )

        if encoder_outputs is None:

            encoder_processing_inputs = {
                "func": self.encoder.call,
                "config": self.encoder.config,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "training": training,
                "kwargs_call": kwargs_encoder,
            }

            # Add arguments to encoder from `kwargs_encoder`
            for k, v in kwargs_encoder.items():
                encoder_processing_inputs[k] = v
            kwargs_encoder = {}

            encoder_inputs = input_processing(**encoder_processing_inputs)

            # handle the init case where `dummy_inputs` returns a dict containing `decoder_input_ids`.
            if "decoder_input_ids" in encoder_inputs:
                decoder_input_ids = encoder_inputs.pop("decoder_input_ids")
            # handle the init case where `dummy_inputs` returns a dict containing `decoder_input_ids`.
            if "decoder_attention_mask" in encoder_inputs:
                decoder_attention_mask = encoder_inputs.pop("decoder_attention_mask")

            encoder_outputs = self.encoder(**encoder_inputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        decoder_processing_inputs = {
            "func": self.decoder.call,
            "config": self.decoder.config,
            "input_ids": decoder_input_ids,
            "attention_mask": decoder_attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": attention_mask,
            "inputs_embeds": decoder_inputs_embeds,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "use_cache": use_cache,
            "past_key_values": past_key_values,
            "return_dict": return_dict,
            "training": training,
            "kwargs_call": kwargs_decoder,
        }

        # Add arguments to decoder from `kwargs_decoder`
        for k, v in kwargs_decoder.items():
            decoder_processing_inputs[k] = v
        kwargs_decoder = {}

        decoder_inputs = input_processing(**decoder_processing_inputs)
        decoder_outputs = self.decoder(**decoder_inputs)

        loss = None if decoder_inputs["labels"] is None else decoder_outputs[0]
        logits = decoder_outputs[0] if decoder_inputs["labels"] is None else decoder_outputs[1]
        past_key_values = None

        if decoder_inputs["use_cache"]:
            past_key_values = decoder_outputs[1] if decoder_inputs["labels"] is None else decoder_outputs[2]
        # The starting index of the remaining elements in `decoder_outputs`
        start_index = sum([1 if x is not None else 0 for x in (loss, logits, past_key_values)])

        past = (encoder_outputs[0], past_key_values) if past_key_values else None

        if not decoder_inputs["return_dict"]:
            if not isinstance(encoder_outputs, tuple):
                encoder_outputs = encoder_outputs.to_tuple()
            output = (loss, logits, past) + decoder_outputs[start_index:] + encoder_outputs
            output = tuple([x for x in output if x is not None])
            return output

        return TFSeq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=past,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        cross_attns = (
            tf.convert_to_tensor(output.cross_attentions)
            if self.config.output_attentions and output.cross_attentions is not None
            else None
        )

        return TFSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
            cross_attentions=cross_attns,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past,
        attention_mask,
        use_cache=None,
        **kwargs,
    ):
        if past is None or len(past) not in {1, 2}:
            raise ValueError(f"past has to be an iterable of length 1,2 got {past}")

        if len(past) == 1:
            if not isinstance(past[0], tf.Tensor):
                raise ValueError(f"`past[0]` has to be of type `tf.Tensor`, but is {type(past[0])}")
            encoder_outputs = TFBaseModelOutput(last_hidden_state=past[0])
            past_key_values = None
        else:
            if len(past) != 2:
                raise ValueError(
                    "`past` has to be of length 2 with the encoder_outputs at the first position and past_key_values at the second position."
                )
            encoder_outputs, past_key_values = past
            if isinstance(encoder_outputs, tuple):
                if not isinstance(encoder_outputs[0], tf.Tensor):
                    raise ValueError(
                        f"`encoder_outputs[0]` has to be of type `tf.Tensor`, but is {type(encoder_outputs[0])}"
                    )
                encoder_outputs = TFBaseModelOutput(last_hidden_state=encoder_outputs[0])
            elif isinstance(encoder_outputs, tf.Tensor):
                encoder_outputs = TFBaseModelOutput(last_hidden_state=encoder_outputs)
            if not past_key_values:
                raise ValueError(
                    f"decoder cached states must be truthy. got {past_key_values} from the 2nd element of past"
                )
            decoder_input_ids = decoder_input_ids[:, -1:]

        if not isinstance(encoder_outputs, TFBaseModelOutput):
            raise ValueError(f"encoder_outputs should be a TFBaseModelOutput, Instead got {type(encoder_outputs)}.")

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the TFEncoderDecoderModel directly is not supported."
            "Please use the respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        if len(past) == 1:
            return past

        encoder_outputs, past_key_values = past

        return (encoder_outputs, self.decoder._reorder_cache(past_key_values, beam_idx))
