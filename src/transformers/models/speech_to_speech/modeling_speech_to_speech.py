# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
""" Classes to support Speech-Encoder-Text-Decoder architectures"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d, CrossEntropyLoss

from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
from .configuration_hifigan import CodeHiFiGANConfig, HiFiGANConfig
from .configuration_speech_to_speech import SpeechToSpeechConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "SpeechToSpeechConfig"

SPEECH_ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a speech-sequence-to-text-sequence model with any pretrained speech
    autoencoding model as the encoder and any pretrained text autoregressive model as the decoder. The encoder is
    loaded via [`~AutoModel.from_pretrained`] function and the decoder is loaded via
    [`~AutoModelForCausalLM.from_pretrained`] function. Cross-attention layers are automatically added to the decoder
    and should be fine-tuned on a downstream generative task, like summarization.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    Additionally, in [Large-Scale Self- and Semi-Supervised Learning for Speech
    Translation](https://arxiv.org/abs/2104.06678) it is shown how leveraging large pretrained speech models for speech
    translation yields a significant performance improvement.

    After such an Speech-Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other
    models (see the examples for more information).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechToSpeechConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SPEECH_ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, feature_dim)`, *optional*):
            Float values of input raw speech waveform or speech features. Values can be obtained by loading a *.flac*
            or *.wav* audio file into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile
            library (*pip install soundfile*). To prepare the array into *inputs*, either the [`Wav2Vec2Processor`] or
            [`Speech2TextProcessor`] should be used for padding and conversion into a tensor of type
            *torch.FloatTensor*.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            For training, `decoder_input_ids` are automatically created by the model by shifting the `labels` to the
            right, replacing -100 by the `pad_token_id` and prepending them with the `decoder_start_token_id`.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        encoder_outputs (`tuple(torch.FloatTensor)`, *optional*):
            This tuple must consist of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) is a tensor
            of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the
            decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert `decoder_input_ids` indices
            into associated vectors than the model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss for the decoder. Indices should be in `[-100, 0,
            ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`Wav2Vec2Processor`] should be used for padding
            and conversion into a tensor of type *torch.FloatTensor*. See [`Wav2Vec2Processor.__call__`] for details.
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_size)`, *optional*):
            Float values of fbank features extracted from the raw speech waveform. Raw speech waveform can be obtained
            by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.*
            via the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`Speech2TextFeatureExtractor`] should be used for extracting the fbank features, padding and conversion
            into a tensor of type `torch.FloatTensor`. See [`~Speech2TextFeatureExtractor.__call__`]
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.Seq2SeqLMOutput`] instead of a plain tuple.
        kwargs: (*optional*) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:

            - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.
            - With a *decoder_* prefix which will be input as `**decoder_kwargs` for the decoder forward function.
"""


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@add_start_docstrings(SPEECH_ENCODER_DECODER_START_DOCSTRING)
# Copied from transformers.models.speech_encoder_decoder.modeling_speech_encoder_decoder.SpeechEncoderDecoderModel with SpeechEncoderDecoder->SpeechToSpeech,speech_encoder_decoder->speech_to_speech
class SpeechToSpeechModel(PreTrainedModel):
    r"""
    [`SpeechToSpeechModel`] is a generic model class that will be instantiated as a transformer architecture with one
    of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = SpeechToSpeechConfig
    base_model_prefix = "speech_to_speech"
    main_input_name = "inputs"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = SpeechToSpeechConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # get encoder output hidden size
        self.encoder_output_dim = getattr(config.encoder, "output_hidden_size", config.encoder.hidden_size)
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            # encoder outputs might need to be projected to different dimension for decoder
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

    def _set_gradient_checkpointing(self, module, value=False):
        # call both encoder and decoder function on gradient checkpointing
        self.encoder._set_gradient_checkpointing(module, value=value)
        self.decoder._set_gradient_checkpointing(module, value=value)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder of the speech encoder so
        that its parameters will not be updated during training.
        """
        self.encoder.freeze_feature_encoder()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for SpeechToSpeechModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

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
        >>> from transformers import SpeechToSpeechModel

        >>> # initialize a wav2vec2bert from a pretrained Wav2Vec2 and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized
        >>> model = SpeechToSpeechModel.from_encoder_decoder_pretrained(
        ...     "facebook/wav2vec2-base-960h", "bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./wav2vec2bert")
        >>> # load fine-tuned model
        >>> model = SpeechToSpeechModel.from_pretrained("./wav2vec2bert")
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
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
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

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = SpeechToSpeechConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    @add_start_docstrings_to_model_forward(SPEECH_ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        inputs=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        input_values=None,
        input_features=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import SpeechToSpeechModel, Wav2Vec2Processor
        >>> from datasets import load_dataset
        >>> import torch

        >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")
        >>> model = SpeechToSpeechModel.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values
        >>> # Inference: Translate English speech to German
        >>> generated = model.generate(input_values)
        >>> decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
        >>> decoded
        'Mr. Quilter ist der Apostel der Mittelschicht und wir freuen uns, sein Evangelium willkommen heißen zu können.'

        >>> # Training: Train model on English transcription
        >>> with processor.as_target_processor():
        ...     labels = processor(ds[0]["text"], return_tensors="pt").input_ids

        >>> loss = model(input_values, labels=labels).loss
        >>> loss.backward()
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if inputs is None:
                if input_values is not None and input_features is not None:
                    raise ValueError("You cannot specify both input_values and input_features at the same time")
                elif input_values is not None:
                    inputs = input_values
                elif input_features is not None:
                    inputs = input_features
                else:
                    raise ValueError("You have to specify either input_values or input_features")

            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # compute correct encoder attention mask
        if attention_mask is not None:
            encoder_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_hidden_states.shape[1], attention_mask
            )
        else:
            encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the SpeechToSpeechModel directly is not supported. Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)


class HiFiGANResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super(HiFiGANResidualBlock, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )
        self.leaky_relu_slope = leaky_relu_slope

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = F.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = F.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


class HiFiGANModel(PreTrainedModel):
    config_class = HiFiGANConfig

    def __init__(self, config: HiFiGANConfig):
        super().__init__(config)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = Conv1d(
            config.model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HiFiGANResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        self.conv_post = Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, hidden_states):
        hidden_states = self.conv_pre(hidden_states)
        for i in range(self.num_upsamples):
            hidden_states = F.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = F.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        return hidden_states


class CodeHiFiGANVariancePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(
            config.encoder_embed_dim,
            config.variance_predictor_hidden_dim,
            kernel_size=config.variance_predictor_kernel_size,
            padding=(config.variance_predictor_kernel_size - 1) // 2,
        )
        self.ln1 = nn.LayerNorm(config.variance_predictor_hidden_dim)
        self.dropout_module = nn.Dropout(p=config.variance_predictor_dropout)
        self.conv2 = nn.Conv1d(
            config.variance_predictor_hidden_dim,
            config.variance_predictor_hidden_dim,
            kernel_size=config.variance_predictor_kernel_size,
            padding=1,
        )
        self.ln2 = nn.LayerNorm(config.variance_predictor_hidden_dim)
        self.proj = nn.Linear(config.variance_predictor_hidden_dim, 1)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, sequence_length, channels)
        hidden_states = F.relu(self.conv1(hidden_states.transpose(1, 2))).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln1(hidden_states))
        hidden_states = F.relu(self.conv2(hidden_states.transpose(1, 2))).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln2(hidden_states))
        out = self.proj(hidden_states).squeeze(dim=2)
        # out: (batch_size, sequence_length)
        return out


class CodeHiFiGANModel(HiFiGANModel):
    config_class = CodeHiFiGANConfig

    def __init__(self, config):
        super().__init__(config)
        self.dict = nn.Embedding(config.num_embeddings, config.embedding_dim)
        self.multispeaker = config.multispeaker
        self.speaker_embedding = config.speaker_embedding

        if self.multispeaker and not self.speaker_embedding:
            self.speaker = nn.Embedding(config.num_speakers, config.embedding_dim)
        elif self.speaker_embedding:
            self.speaker = nn.Linear(config.speaker_embedding_dim, config.embedding_dim)

        self.duration_predictor = None
        if config.duration_predictor:
            self.duration_predictor = CodeHiFiGANVariancePredictor(config)

        self.f0 = config.f0
        n_f0_bin = config.f0_quant_num_bin

        self.f0_quant_embed = None
        if n_f0_bin > 0:
            self.f0_quant_embed = nn.Embedding(n_f0_bin, config.embedding_dim)

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            batch_size, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            batch_size, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            batch_size, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        remainder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if remainder > 0:
            raise NotImplementedError("Padding condition signal - misalignment between condition features.")

        signal = signal.view(batch_size, channels, max_frames)
        return signal

    def forward(self, hidden_states, duration_prediction=False, f0=None, speaker=None):
        hidden_states = self.dict(hidden_states).transpose(1, 2)

        if self.duration_predictor and duration_prediction:
            # TODO: handle batched inputs
            assert hidden_states.size(0) == 1, "only support single sample"
            log_dur_pred = self.duration_predictor(hidden_states.transpose(1, 2))
            dur_out = torch.clamp(torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1)
            # hidden_states: (batch_size, channels, sequence_length)
            hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)

        if self.f0:
            assert f0 is not None, 'require "f0" if "config.f0" is True'
            if self.f0_quant_embed:
                f0 = self.f0_quant_embed(f0.long()).transpose(1, 2)
            else:
                f0 = f0.unsqueeze(1)

            if hidden_states.shape[-1] < f0.shape[-1]:
                hidden_states = self._upsample(hidden_states, f0.shape[-1])
            elif hidden_states.shape[-1] > f0.shape[-1]:
                f0 = self._upsample(f0, hidden_states.shape[-1])
            hidden_states = torch.cat([hidden_states, f0], dim=1)

        if self.multispeaker:
            assert speaker is not None, 'require "speaker" input for multispeaker CodeHiFiGAN vocoder'
            speaker = self.speaker(speaker).transpose(1, 2)
            speaker = self._upsample(speaker, hidden_states.shape[-1])
            hidden_states = torch.cat([hidden_states, speaker], dim=1)

        return super().forward(hidden_states)


class SpeechToSpeechModelWithCodeHiFiGAN(SpeechToSpeechModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = SpeechToSpeechConfig,
        s2ut: Optional[PreTrainedModel] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        vocoder: Optional[PreTrainedModel] = None,
        vocoder_config: Optional[PretrainedConfig] = CodeHiFiGANConfig,
        **kwargs
    ):

        super().__init__(config)

        encoder = encoder if encoder is not None else s2ut.encoder
        decoder = decoder if decoder is not None else s2ut.decoder

        self.model = SpeechToSpeechModel(config, encoder=encoder, decoder=decoder)

        if vocoder is None:
            vocoder = CodeHiFiGANModel(vocoder_config)

        self.vocoder = vocoder

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_vocoder(self):
        return self.vocoder

    def get_output_embeddings(self):
        return self.model.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.model.decoder.set_output_embeddings(new_embeddings)

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder of the speech encoder so
        that its parameters will not be updated during training.
        """
        self.model.encoder.freeze_feature_encoder()

    @classmethod
    def from_s2ut_vocoder_pretrained(
        cls,
        s2ut_pretrained_model_name_or_path: str = None,
        vocoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r"""
        Instantiate an S2UT and vocoder from one or two base classes of the library from pretrained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            s2ut_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the S2UT model. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            vecoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the vocoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the vocoder configuration, use the prefix *vocoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import SpeechToSpeechModelWithCodeHiFiGAN

        >>> # initialize a SpeechToSpeech model with a vocoder from a pretrained S2UT and a pretrained vocoder model
        >>> model = SpeechToSpeechModelWithCodeHiFiGAN.from_s2ut_vocoder_pretrained(
        ...     "facebook/s2ut-es-to-en", "facebook/codehifigan"  # TODO: checkpoint names
        ... )
        >>> # saving model after loading
        >>> model.save_pretrained("./s2st-es-to-en")
        >>> # load fine-tuned model
        >>> model = SpeechToSpeechModel.from_pretrained("./s2st-es-to-en")
        ```"""

        kwargs_s2ut = {
            argument[len("s2ut_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_vocoder = {
            argument[len("vocoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove s2ut, vocoder kwargs from kwargs
        for key in kwargs_s2ut.keys():
            del kwargs["s2ut_" + key]
        for key in kwargs_vocoder.keys():
            del kwargs["vocoder_" + key]

        # Load and initialize the s2ut and vocoder
        s2ut = kwargs_s2ut.pop("model", None)
        if s2ut is None:
            if s2ut_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `s2ut_model` is not defined as an argument, a `s2ut_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_s2ut:
                s2ut_config, kwargs_s2ut = SpeechToSpeechConfig.from_pretrained(
                    s2ut_pretrained_model_name_or_path, **kwargs_s2ut, return_unused_kwargs=True
                )

                kwargs_s2ut["config"] = s2ut_config

            s2ut = SpeechToSpeechModel.from_pretrained(s2ut_pretrained_model_name_or_path, *model_args, **kwargs_s2ut)

        vocoder = kwargs_vocoder.pop("model", None)
        if vocoder is None:
            if vocoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `vocoder_model` is not defined as an argument, a `vocoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_vocoder:
                # TODO: AutoConfig for vocoder
                vocoder_config, kwargs_vocoder = CodeHiFiGANConfig.from_pretrained(
                    vocoder_pretrained_model_name_or_path, **kwargs_vocoder, return_unused_kwargs=True
                )
                kwargs_vocoder["config"] = vocoder_config

            # TODO: AutoModel for vocoder
            vocoder = CodeHiFiGANModel.from_pretrained(vocoder_pretrained_model_name_or_path, **kwargs_vocoder)

        # instantiate config with corresponding kwargs
        config = SpeechToSpeechConfig.from_encoder_decoder_configs(s2ut.config.encoder, s2ut.config.decoder, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(config=config, s2ut=s2ut, vocoder=vocoder)

    def generate(self, **kwargs):
        model_out = self.model.generate(**kwargs)

        vocoder_in = [label_to_id[int(i)] for i in model_out[0]]
        # skip 'special' tokens
        vocoder_in = [i for i in vocoder_in if type(i) != str]

        vocoder_in = torch.tensor(vocoder_in).unsqueeze(0)
        vocoder_out = self.vocoder(vocoder_in, duration_prediction=True)

        return vocoder_out


# 6 labels map to special tokens
# the remainder are subtracted by 4 for each index
label_to_id = {
    0: "<s>",
    1: "<pad>",
    2: "</s>",
    3: "<unk>",
    4: 0,
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 5,
    10: 6,
    11: 7,
    12: 8,
    13: 9,
    14: 10,
    15: 11,
    16: 12,
    17: 13,
    18: 14,
    19: 15,
    20: 16,
    21: 17,
    22: 18,
    23: 19,
    24: 20,
    25: 21,
    26: 22,
    27: 23,
    28: 24,
    29: 25,
    30: 26,
    31: 27,
    32: 28,
    33: 29,
    34: 30,
    35: 31,
    36: 32,
    37: 33,
    38: 34,
    39: 35,
    40: 36,
    41: 37,
    42: 38,
    43: 39,
    44: 40,
    45: 41,
    46: 42,
    47: 43,
    48: 44,
    49: 45,
    50: 46,
    51: 47,
    52: 48,
    53: 49,
    54: 50,
    55: 51,
    56: 52,
    57: 53,
    58: 54,
    59: 55,
    60: 56,
    61: 57,
    62: 58,
    63: 59,
    64: 60,
    65: 61,
    66: 62,
    67: 63,
    68: 64,
    69: 65,
    70: 66,
    71: 67,
    72: 68,
    73: 69,
    74: 70,
    75: 71,
    76: 72,
    77: 73,
    78: 74,
    79: 75,
    80: 76,
    81: 77,
    82: 78,
    83: 79,
    84: 80,
    85: 81,
    86: 82,
    87: 83,
    88: 84,
    89: 85,
    90: 86,
    91: 87,
    92: 88,
    93: 89,
    94: 90,
    95: 91,
    96: 92,
    97: 93,
    98: 94,
    99: 95,
    100: 96,
    101: 97,
    102: 98,
    103: 99,
    104: 100,
    105: 101,
    106: 102,
    107: 103,
    108: 104,
    109: 105,
    110: 106,
    111: 107,
    112: 108,
    113: 109,
    114: 110,
    115: 111,
    116: 112,
    117: 113,
    118: 114,
    119: 115,
    120: 116,
    121: 117,
    122: 118,
    123: 119,
    124: 120,
    125: 121,
    126: 122,
    127: 123,
    128: 124,
    129: 125,
    130: 126,
    131: 127,
    132: 128,
    133: 129,
    134: 130,
    135: 131,
    136: 132,
    137: 133,
    138: 134,
    139: 135,
    140: 136,
    141: 137,
    142: 138,
    143: 139,
    144: 140,
    145: 141,
    146: 142,
    147: 143,
    148: 144,
    149: 145,
    150: 146,
    151: 147,
    152: 148,
    153: 149,
    154: 150,
    155: 151,
    156: 152,
    157: 153,
    158: 154,
    159: 155,
    160: 156,
    161: 157,
    162: 158,
    163: 159,
    164: 160,
    165: 161,
    166: 162,
    167: 163,
    168: 164,
    169: 165,
    170: 166,
    171: 167,
    172: 168,
    173: 169,
    174: 170,
    175: 171,
    176: 172,
    177: 173,
    178: 174,
    179: 175,
    180: 176,
    181: 177,
    182: 178,
    183: 179,
    184: 180,
    185: 181,
    186: 182,
    187: 183,
    188: 184,
    189: 185,
    190: 186,
    191: 187,
    192: 188,
    193: 189,
    194: 190,
    195: 191,
    196: 192,
    197: 193,
    198: 194,
    199: 195,
    200: 196,
    201: 197,
    202: 198,
    203: 199,
    204: 200,
    205: 201,
    206: 202,
    207: 203,
    208: 204,
    209: 205,
    210: 206,
    211: 207,
    212: 208,
    213: 209,
    214: 210,
    215: 211,
    216: 212,
    217: 213,
    218: 214,
    219: 215,
    220: 216,
    221: 217,
    222: 218,
    223: 219,
    224: 220,
    225: 221,
    226: 222,
    227: 223,
    228: 224,
    229: 225,
    230: 226,
    231: 227,
    232: 228,
    233: 229,
    234: 230,
    235: 231,
    236: 232,
    237: 233,
    238: 234,
    239: 235,
    240: 236,
    241: 237,
    242: 238,
    243: 239,
    244: 240,
    245: 241,
    246: 242,
    247: 243,
    248: 244,
    249: 245,
    250: 246,
    251: 247,
    252: 248,
    253: 249,
    254: 250,
    255: 251,
    256: 252,
    257: 253,
    258: 254,
    259: 255,
    260: 256,
    261: 257,
    262: 258,
    263: 259,
    264: 260,
    265: 261,
    266: 262,
    267: 263,
    268: 264,
    269: 265,
    270: 266,
    271: 267,
    272: 268,
    273: 269,
    274: 270,
    275: 271,
    276: 272,
    277: 273,
    278: 274,
    279: 275,
    280: 276,
    281: 277,
    282: 278,
    283: 279,
    284: 280,
    285: 281,
    286: 282,
    287: 283,
    288: 284,
    289: 285,
    290: 286,
    291: 287,
    292: 288,
    293: 289,
    294: 290,
    295: 291,
    296: 292,
    297: 293,
    298: 294,
    299: 295,
    300: 296,
    301: 297,
    302: 298,
    303: 299,
    304: 300,
    305: 301,
    306: 302,
    307: 303,
    308: 304,
    309: 305,
    310: 306,
    311: 307,
    312: 308,
    313: 309,
    314: 310,
    315: 311,
    316: 312,
    317: 313,
    318: 314,
    319: 315,
    320: 316,
    321: 317,
    322: 318,
    323: 319,
    324: 320,
    325: 321,
    326: 322,
    327: 323,
    328: 324,
    329: 325,
    330: 326,
    331: 327,
    332: 328,
    333: 329,
    334: 330,
    335: 331,
    336: 332,
    337: 333,
    338: 334,
    339: 335,
    340: 336,
    341: 337,
    342: 338,
    343: 339,
    344: 340,
    345: 341,
    346: 342,
    347: 343,
    348: 344,
    349: 345,
    350: 346,
    351: 347,
    352: 348,
    353: 349,
    354: 350,
    355: 351,
    356: 352,
    357: 353,
    358: 354,
    359: 355,
    360: 356,
    361: 357,
    362: 358,
    363: 359,
    364: 360,
    365: 361,
    366: 362,
    367: 363,
    368: 364,
    369: 365,
    370: 366,
    371: 367,
    372: 368,
    373: 369,
    374: 370,
    375: 371,
    376: 372,
    377: 373,
    378: 374,
    379: 375,
    380: 376,
    381: 377,
    382: 378,
    383: 379,
    384: 380,
    385: 381,
    386: 382,
    387: 383,
    388: 384,
    389: 385,
    390: 386,
    391: 387,
    392: 388,
    393: 389,
    394: 390,
    395: 391,
    396: 392,
    397: 393,
    398: 394,
    399: 395,
    400: 396,
    401: 397,
    402: 398,
    403: 399,
    404: 400,
    405: 401,
    406: 402,
    407: 403,
    408: 404,
    409: 405,
    410: 406,
    411: 407,
    412: 408,
    413: 409,
    414: 410,
    415: 411,
    416: 412,
    417: 413,
    418: 414,
    419: 415,
    420: 416,
    421: 417,
    422: 418,
    423: 419,
    424: 420,
    425: 421,
    426: 422,
    427: 423,
    428: 424,
    429: 425,
    430: 426,
    431: 427,
    432: 428,
    433: 429,
    434: 430,
    435: 431,
    436: 432,
    437: 433,
    438: 434,
    439: 435,
    440: 436,
    441: 437,
    442: 438,
    443: 439,
    444: 440,
    445: 441,
    446: 442,
    447: 443,
    448: 444,
    449: 445,
    450: 446,
    451: 447,
    452: 448,
    453: 449,
    454: 450,
    455: 451,
    456: 452,
    457: 453,
    458: 454,
    459: 455,
    460: 456,
    461: 457,
    462: 458,
    463: 459,
    464: 460,
    465: 461,
    466: 462,
    467: 463,
    468: 464,
    469: 465,
    470: 466,
    471: 467,
    472: 468,
    473: 469,
    474: 470,
    475: 471,
    476: 472,
    477: 473,
    478: 474,
    479: 475,
    480: 476,
    481: 477,
    482: 478,
    483: 479,
    484: 480,
    485: 481,
    486: 482,
    487: 483,
    488: 484,
    489: 485,
    490: 486,
    491: 487,
    492: 488,
    493: 489,
    494: 490,
    495: 491,
    496: 492,
    497: 493,
    498: 494,
    499: 495,
    500: 496,
    501: 497,
    502: 498,
    503: 499,
    504: 500,
    505: 501,
    506: 502,
    507: 503,
    508: 504,
    509: 505,
    510: 506,
    511: 507,
    512: 508,
    513: 509,
    514: 510,
    515: 511,
    516: 512,
    517: 513,
    518: 514,
    519: 515,
    520: 516,
    521: 517,
    522: 518,
    523: 519,
    524: 520,
    525: 521,
    526: 522,
    527: 523,
    528: 524,
    529: 525,
    530: 526,
    531: 527,
    532: 528,
    533: 529,
    534: 530,
    535: 531,
    536: 532,
    537: 533,
    538: 534,
    539: 535,
    540: 536,
    541: 537,
    542: 538,
    543: 539,
    544: 540,
    545: 541,
    546: 542,
    547: 543,
    548: 544,
    549: 545,
    550: 546,
    551: 547,
    552: 548,
    553: 549,
    554: 550,
    555: 551,
    556: 552,
    557: 553,
    558: 554,
    559: 555,
    560: 556,
    561: 557,
    562: 558,
    563: 559,
    564: 560,
    565: 561,
    566: 562,
    567: 563,
    568: 564,
    569: 565,
    570: 566,
    571: 567,
    572: 568,
    573: 569,
    574: 570,
    575: 571,
    576: 572,
    577: 573,
    578: 574,
    579: 575,
    580: 576,
    581: 577,
    582: 578,
    583: 579,
    584: 580,
    585: 581,
    586: 582,
    587: 583,
    588: 584,
    589: 585,
    590: 586,
    591: 587,
    592: 588,
    593: 589,
    594: 590,
    595: 591,
    596: 592,
    597: 593,
    598: 594,
    599: 595,
    600: 596,
    601: 597,
    602: 598,
    603: 599,
    604: 600,
    605: 601,
    606: 602,
    607: 603,
    608: 604,
    609: 605,
    610: 606,
    611: 607,
    612: 608,
    613: 609,
    614: 610,
    615: 611,
    616: 612,
    617: 613,
    618: 614,
    619: 615,
    620: 616,
    621: 617,
    622: 618,
    623: 619,
    624: 620,
    625: 621,
    626: 622,
    627: 623,
    628: 624,
    629: 625,
    630: 626,
    631: 627,
    632: 628,
    633: 629,
    634: 630,
    635: 631,
    636: 632,
    637: 633,
    638: 634,
    639: 635,
    640: 636,
    641: 637,
    642: 638,
    643: 639,
    644: 640,
    645: 641,
    646: 642,
    647: 643,
    648: 644,
    649: 645,
    650: 646,
    651: 647,
    652: 648,
    653: 649,
    654: 650,
    655: 651,
    656: 652,
    657: 653,
    658: 654,
    659: 655,
    660: 656,
    661: 657,
    662: 658,
    663: 659,
    664: 660,
    665: 661,
    666: 662,
    667: 663,
    668: 664,
    669: 665,
    670: 666,
    671: 667,
    672: 668,
    673: 669,
    674: 670,
    675: 671,
    676: 672,
    677: 673,
    678: 674,
    679: 675,
    680: 676,
    681: 677,
    682: 678,
    683: 679,
    684: 680,
    685: 681,
    686: 682,
    687: 683,
    688: 684,
    689: 685,
    690: 686,
    691: 687,
    692: 688,
    693: 689,
    694: 690,
    695: 691,
    696: 692,
    697: 693,
    698: 694,
    699: 695,
    700: 696,
    701: 697,
    702: 698,
    703: 699,
    704: 700,
    705: 701,
    706: 702,
    707: 703,
    708: 704,
    709: 705,
    710: 706,
    711: 707,
    712: 708,
    713: 709,
    714: 710,
    715: 711,
    716: 712,
    717: 713,
    718: 714,
    719: 715,
    720: 716,
    721: 717,
    722: 718,
    723: 719,
    724: 720,
    725: 721,
    726: 722,
    727: 723,
    728: 724,
    729: 725,
    730: 726,
    731: 727,
    732: 728,
    733: 729,
    734: 730,
    735: 731,
    736: 732,
    737: 733,
    738: 734,
    739: 735,
    740: 736,
    741: 737,
    742: 738,
    743: 739,
    744: 740,
    745: 741,
    746: 742,
    747: 743,
    748: 744,
    749: 745,
    750: 746,
    751: 747,
    752: 748,
    753: 749,
    754: 750,
    755: 751,
    756: 752,
    757: 753,
    758: 754,
    759: 755,
    760: 756,
    761: 757,
    762: 758,
    763: 759,
    764: 760,
    765: 761,
    766: 762,
    767: 763,
    768: 764,
    769: 765,
    770: 766,
    771: 767,
    772: 768,
    773: 769,
    774: 770,
    775: 771,
    776: 772,
    777: 773,
    778: 774,
    779: 775,
    780: 776,
    781: 777,
    782: 778,
    783: 779,
    784: 780,
    785: 781,
    786: 782,
    787: 783,
    788: 784,
    789: 785,
    790: 786,
    791: 787,
    792: 788,
    793: 789,
    794: 790,
    795: 791,
    796: 792,
    797: 793,
    798: 794,
    799: 795,
    800: 796,
    801: 797,
    802: 798,
    803: 799,
    804: 800,
    805: 801,
    806: 802,
    807: 803,
    808: 804,
    809: 805,
    810: 806,
    811: 807,
    812: 808,
    813: 809,
    814: 810,
    815: 811,
    816: 812,
    817: 813,
    818: 814,
    819: 815,
    820: 816,
    821: 817,
    822: 818,
    823: 819,
    824: 820,
    825: 821,
    826: 822,
    827: 823,
    828: 824,
    829: 825,
    830: 826,
    831: 827,
    832: 828,
    833: 829,
    834: 830,
    835: 831,
    836: 832,
    837: 833,
    838: 834,
    839: 835,
    840: 836,
    841: 837,
    842: 838,
    843: 839,
    844: 840,
    845: 841,
    846: 842,
    847: 843,
    848: 844,
    849: 845,
    850: 846,
    851: 847,
    852: 848,
    853: 849,
    854: 850,
    855: 851,
    856: 852,
    857: 853,
    858: 854,
    859: 855,
    860: 856,
    861: 857,
    862: 858,
    863: 859,
    864: 860,
    865: 861,
    866: 862,
    867: 863,
    868: 864,
    869: 865,
    870: 866,
    871: 867,
    872: 868,
    873: 869,
    874: 870,
    875: 871,
    876: 872,
    877: 873,
    878: 874,
    879: 875,
    880: 876,
    881: 877,
    882: 878,
    883: 879,
    884: 880,
    885: 881,
    886: 882,
    887: 883,
    888: 884,
    889: 885,
    890: 886,
    891: 887,
    892: 888,
    893: 889,
    894: 890,
    895: 891,
    896: 892,
    897: 893,
    898: 894,
    899: 895,
    900: 896,
    901: 897,
    902: 898,
    903: 899,
    904: 900,
    905: 901,
    906: 902,
    907: 903,
    908: 904,
    909: 905,
    910: 906,
    911: 907,
    912: 908,
    913: 909,
    914: 910,
    915: 911,
    916: 912,
    917: 913,
    918: 914,
    919: 915,
    920: 916,
    921: 917,
    922: 918,
    923: 919,
    924: 920,
    925: 921,
    926: 922,
    927: 923,
    928: 924,
    929: 925,
    930: 926,
    931: 927,
    932: 928,
    933: 929,
    934: 930,
    935: 931,
    936: 932,
    937: 933,
    938: 934,
    939: 935,
    940: 936,
    941: 937,
    942: 938,
    943: 939,
    944: 940,
    945: 941,
    946: 942,
    947: 943,
    948: 944,
    949: 945,
    950: 946,
    951: 947,
    952: 948,
    953: 949,
    954: 950,
    955: 951,
    956: 952,
    957: 953,
    958: 954,
    959: 955,
    960: 956,
    961: 957,
    962: 958,
    963: 959,
    964: 960,
    965: 961,
    966: 962,
    967: 963,
    968: 964,
    969: 965,
    970: 966,
    971: 967,
    972: 968,
    973: 969,
    974: 970,
    975: 971,
    976: 972,
    977: 973,
    978: 974,
    979: 975,
    980: 976,
    981: 977,
    982: 978,
    983: 979,
    984: 980,
    985: 981,
    986: 982,
    987: 983,
    988: 984,
    989: 985,
    990: 986,
    991: 987,
    992: 988,
    993: 989,
    994: 990,
    995: 991,
    996: 992,
    997: 993,
    998: 994,
    999: 995,
    1000: 996,
    1001: 997,
    1002: 998,
    1003: 999,
    1004: "<lang:en>",
    1005: "<lang:es>",
    1006: "<mask>",
}
