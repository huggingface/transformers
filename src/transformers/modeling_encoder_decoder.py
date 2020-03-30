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
import os

from .modeling_auto import AutoModel, AutoModelWithLMHead
from .modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)


class EncoderDecoderModel(PreTrainedModel):
    r"""
        :class:`~transformers.EncoderDecoder` is a generic model class that will be
        instantiated as a transformer architecture with one of the base model
        classes of the library as encoder and another one as
        decoder when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method for the encoder and `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` class method for the decoder.
    """

    def __init__(self, encoder, decoder):
        assert encoder is not None, "The encoder has to be defined"
        assert decoder is not None, "The decoder has to be defined"

        config = self._init_config(encoder.config, decoder.config)
        super().__init__(config)

        self.encoder = encoder
        assert (
            self.encoder.get_output_embeddings() is None
        ), "The encoder {} should not have a LM Head. Please use a model without LM Head"
        self.decoder = decoder

    def _init_config(self, encoder_config, decoder_config):
        # decoder config is used as default config (important for generation)
        # TODO: check with thom
        config = decoder_config
        config.is_encoder_decoder = True

        return config

    def get_encoder(self):
        return self.encoder

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path=None, decoder_pretrained_model_name_or_path=None, *model_args, **kwargs
    ):
        r""" Instantiates an encoder and a decoder from one or two base classes of the library from pre-trained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you need to first set it back in training mode with `model.train()`

        Params:
            encoder_pretrained_model_name_or_path: information necessary to initiate the encoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/encoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path: information necessary to initiate the decoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/decoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments.
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

                You can specify kwargs sepcific for the encoder and decoder by prefixing the key with `encoder_` and `decoder_` respectively. (e.g. ``decoder_output_attention=True``). The remaining kwargs will be passed to both encoders and decoders.

        Examples::

            # For example purposes. Not runnable.
            model = PreTrainedEncoderDecoder.from_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert
        """

        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as a whole.
        # We let the specific kwargs override the common ones in case of conflict.

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("encoder_model", None)
        if encoder is None:
            assert (
                pretrained_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"
            encoder = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs_encoder)
        encoder.config.is_decoder = False

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            assert (
                decoder_pretrained_model_name_or_path is not None
            ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"
            decoder = AutoModelWithLMHead.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
        decoder.config.is_decoder = True

        model = cls(encoder, decoder)

        return model

    def save_pretrained(self, save_directory):
        """ Save a Seq2Seq model and its configuration file in a format such
        that it can be loaded using `:func:`~transformers.PreTrainedEncoderDecoder.from_pretrained`

        We save the encoder' and decoder's parameters in two separate directories.
        """

        # If the root output directory does not exist, create it
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        # Check whether the output directory is empty or not
        sub_directories = [
            directory
            for directory in os.listdir(save_directory)
            if os.path.isdir(os.path.join(save_directory, directory))
        ]

        if len(sub_directories) > 0:
            if "encoder" in sub_directories and "decoder" in sub_directories:
                print(
                    "WARNING: there is an older version of encoder-decoder saved in"
                    + " the output directory. The default behaviour is to overwrite them."
                )

            # Empty the output directory
            for directory_to_remove in sub_directories:
                # Remove all files into the subdirectory
                files_to_remove = os.listdir(os.path.join(save_directory, directory_to_remove))
                for file_to_remove in files_to_remove:
                    os.remove(os.path.join(save_directory, directory_to_remove, file_to_remove))
                # Remove the subdirectory itself
                os.rmdir(os.path.join(save_directory, directory_to_remove))

            assert len(os.listdir(save_directory)) == 0  # sanity check

        # Create the "encoder" directory inside the output directory and save the encoder into it
        if not os.path.exists(os.path.join(save_directory, "encoder")):
            os.mkdir(os.path.join(save_directory, "encoder"))
        self.encoder.save_pretrained(os.path.join(save_directory, "encoder"))

        # Create the "encoder" directory inside the output directory and save the decoder into it
        if not os.path.exists(os.path.join(save_directory, "decoder")):
            os.mkdir(os.path.join(save_directory, "decoder"))
        self.decoder.save_pretrained(os.path.join(save_directory, "decoder"))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
    ):

        """
        Params:
            encoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of encoder input sequence tokens in the vocabulary.
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
        )

        return decoder_outputs + encoder_outputs

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if type(past) is tuple:
            encoder_outputs = past
        else:
            encoder_outputs = (past,)

        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, attention_mask)

        return {
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "attention_mask": decoder_inputs["attention_mask"],
        }

    def _reorder_cache(self, past, beam_idx):
        # as a default encoder-decoder models do not re-order the past. TODO: might
        # have to be updated, e.g. if GPT2 is to be used as a decoder
        return past
