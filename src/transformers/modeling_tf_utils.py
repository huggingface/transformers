# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""TF general model utils."""


import logging
import os

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format

from .configuration_utils import PretrainedConfig
from .file_utils import DUMMY_INPUTS, TF2_WEIGHTS_NAME, WEIGHTS_NAME, cached_path, hf_bucket_url, is_remote_url
from .modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model


logger = logging.getLogger(__name__)


class TFModelUtilsMixin:
    """
    A few utilities for `tf.keras.Model`s, to be used as a mixin.
    """

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Get number of (optionally, trainable) parameters in the model.
        """
        if only_trainable:
            return int(sum(np.prod(w.shape.as_list()) for w in self.trainable_variables))
        else:
            return self.count_params()


class TFPreTrainedModel(tf.keras.Model, TFModelUtilsMixin):
    r""" Base class for all TF models.

        :class:`~transformers.TFPreTrainedModel` takes care of storing the configuration of the models and handles methods for loading/downloading/saving models
        as well as a few methods common to all models to (i) resize the input embeddings and (ii) prune heads in the self-attention heads.

        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~transformers.PretrainedConfig` to use as configuration class for this model architecture.
            - ``pretrained_model_archive_map``: a python ``dict`` of with `short-cut-names` (string) as keys and `url` (string) of associated pretrained weights as values.
            - ``load_tf_weights``: a python ``method`` for loading a TensorFlow checkpoint in a PyTorch model, taking as arguments:

                - ``model``: an instance of the relevant subclass of :class:`~transformers.PreTrainedModel`,
                - ``config``: an instance of the relevant subclass of :class:`~transformers.PretrainedConfig`,
                - ``path``: a path (string) to the TensorFlow checkpoint.

            - ``base_model_prefix``: a string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
    """
    config_class = None
    pretrained_model_archive_map = {}
    base_model_prefix = ""

    @property
    def dummy_inputs(self):
        """ Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        return {"input_ids": tf.constant(DUMMY_INPUTS)}

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config

    def get_input_embeddings(self):
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`tf.keras.layers.Layer`:
                A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def get_output_embeddings(self):
        """
        Returns the model's output embeddings.

        Returns:
            :obj:`tf.keras.layers.Layer`:
                A torch module mapping hidden states to vocabulary.
        """
        return None  # Overwrite for models with output embeddings

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Variable from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``tf.Variable``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        # if new_num_tokens is None:
        #     return old_embeddings

        # old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        # if old_num_tokens == new_num_tokens:
        #     return old_embeddings

        # # Build new embeddings
        # new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        # new_embeddings.to(old_embeddings.weight.device)

        # # initialize all new embeddings (in particular added tokens)
        # self._init_weights(new_embeddings)

        # # Copy word embeddings from the previous weights
        # num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        # new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        # return new_embeddings

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end.
                If not provided or None: does nothing and just returns a pointer to the input tokens ``tf.Variable`` Module of the model.

        Return: ``tf.Variable``
            Pointer to the input tokens Embeddings Module of the model
        """
        raise NotImplementedError

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.

            Arguments:

                heads_to_prune: dict with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`).
        """
        raise NotImplementedError

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Save configuration file
        self.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, TF2_WEIGHTS_NAME)
        self.save_weights(output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.

        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch state_dict save file` (e.g. `./pt_model/pytorch_model.bin`). In this case, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the PyTorch checkpoint in a TensorFlow model using the provided conversion scripts and loading the TensorFlow model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) one of:
                    - an instance of a class derived from :class:`~transformers.PretrainedConfig`, or
                    - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained()`
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            from_pt: (`optional`) boolean, default False:
                Load the model weights from a PyTorch state_dict save file (see docstring of pretrained_model_name_or_path argument).

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            # For example purposes. Not runnable.
            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_pt = kwargs.pop("from_pt", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
            elif os.path.isdir(pretrained_model_name_or_path):
                if os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_pt` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME], pretrained_model_name_or_path
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path, postfix=(WEIGHTS_NAME if from_pt else TF2_WEIGHTS_NAME)
                )

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                )
            except EnvironmentError as e:
                if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                    logger.error("Couldn't reach server at '{}' to download pretrained weights.".format(archive_file))
                else:
                    logger.error(
                        "Model name '{}' was not found in model name list ({}). "
                        "We assumed '{}' was a path or url but couldn't find any file "
                        "associated to this path or url.".format(
                            pretrained_model_name_or_path,
                            ", ".join(cls.pretrained_model_archive_map.keys()),
                            archive_file,
                        )
                    )
                raise e
            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if from_pt:
            # Load from a PyTorch checkpoint
            return load_pytorch_checkpoint_in_tf2_model(model, resolved_archive_file, allow_missing_keys=True)

        model(model.dummy_inputs, training=False)  # build the network with dummy inputs

        assert os.path.isfile(resolved_archive_file), "Error retrieving file {}".format(resolved_archive_file)
        # 'by_name' allow us to do transfer learning by skipping/adding layers
        # see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1339-L1357
        try:
            model.load_weights(resolved_archive_file, by_name=True)
        except OSError:
            raise OSError(
                "Unable to load weights from h5 file. "
                "If you tried to load a TF 2.0 model from a PyTorch checkpoint, please set from_pt=True. "
            )

        model(model.dummy_inputs, training=False)  # Make sure restore ops are run

        # Check if the models are the same to output loading informations
        with h5py.File(resolved_archive_file, "r") as f:
            if "layer_names" not in f.attrs and "model_weights" in f:
                f = f["model_weights"]
            hdf5_layer_names = set(hdf5_format.load_attributes_from_hdf5_group(f, "layer_names"))
        model_layer_names = set(layer.name for layer in model.layers)
        missing_keys = list(model_layer_names - hdf5_layer_names)
        unexpected_keys = list(hdf5_layer_names - model_layer_names)
        error_msgs = []

        if len(missing_keys) > 0:
            logger.info(
                "Layers of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys)
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Layers from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys)
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading weights for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
            )
        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model

    def prepare_inputs_for_generation(self, inputs, **kwargs):
        return {"inputs": inputs}

    def _do_output_past(self, outputs):
        has_output_past = hasattr(self.config, "output_past") and self.config.output_past
        has_mem_len = hasattr(self.config, "mem_len") and self.config.mem_len

        if has_output_past and not has_mem_len and len(outputs) > 1:
            return True
        elif has_mem_len and self.config.mem_len > 0 and len(outputs) > 1:
            return True

        return False

    def generate(
        self,
        input_ids=None,
        max_length=None,
        do_sample=True,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_ids=None,
        length_penalty=None,
        num_return_sequences=None,
    ):
        r""" Generates sequences for models with a LM head. The method currently supports greedy or penalized greedy decoding, sampling with top-k or nucleus sampling
        and beam-search.

        Adapted in part from `Facebook's XLM beam search code`_.

        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


        Parameters:

            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.

            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between 1 and infinity. Default to 20.

            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `True`.

            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictely positive. Default to 1.0.

            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

            bos_token_id: (`optional`) int
                Beginning of sentence token if no prompt is provided. Default to 0.

            eos_token_ids: (`optional`) int or list of int
                End of sequence token or list of tokens to stop the generation. Default to 0.
            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.

            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.

        Return:

            output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
                sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40, bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id, do_sample=False)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.pad_token_id, eos_token_ids=tokenizer.eos_token_id, num_return_sequences=3)  # 3 generate sequences using by sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_ids = eos_token_ids if eos_token_ids is not None else self.config.eos_token_ids
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        if input_ids is not None:
            batch_size = shape_list(input_ids)[0]  # overriden by the input batch_size
        else:
            batch_size = 1
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
        assert temperature > 0, "`temperature` should be strictely positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_ids is None) or (
            isinstance(eos_token_ids, (list, tuple)) and ((isinstance(e, int) and e >= 0) for e in eos_token_ids)
        ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert length_penalty > 0, "`length_penalty` should be strictely positive."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictely positive integer."

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = tf.fill((batch_size, 1), bos_token_id)
        else:
            assert len(shape_list(input_ids)) == 2, "Input prompt should be of shape (batch_size, sequence length)."

        if pad_token_id is None and eos_token_ids is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_ids[0])
            )
            pad_token_id = eos_token_ids[0]

        # current position and vocab size
        cur_len = shape_list(input_ids)[1]
        vocab_size = self.config.vocab_size

        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = tf.broadcast_to(tf.expand_dims(input_ids, 1), (batch_size, num_return_sequences, cur_len))
            effective_batch_size = batch_size * num_return_sequences
            input_ids = tf.reshape(input_ids, (effective_batch_size, cur_len))
        else:
            effective_batch_size = batch_size

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """

        def _create_next_token_logits_penalties(input_ids, logits):
            # create logit penalties for already seen input_ids
            token_penalties = np.ones(shape_list(logits))
            prev_input_ids = [np.unique(input_id) for input_id in input_ids.numpy()]
            for i, prev_input_id in enumerate(prev_input_ids):
                logit_penalized = logits[i].numpy()[prev_input_id]
                # if previous logit score is < 0 then multiply repetition penalty else divide
                logit_penalized[logit_penalized < 0] = repetition_penalty
                logit_penalized[logit_penalized > 0] = 1 / repetition_penalty
                np.put(token_penalties[i], prev_input_id, logit_penalized)
            return tf.convert_to_tensor(token_penalties, dtype=tf.float32)

        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = tf.ones_like(input_ids[:, 0])
        sent_lengths = tf.ones_like(input_ids[:, 0]) * max_length

        past = None

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                next_token_logits_penalties = _create_next_token_logits_penalties(input_ids, next_token_logits)
                next_token_logits = tf.math.multiply(next_token_logits, next_token_logits_penalties)

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = tf_top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = tf.squeeze(
                    tf.random.categorical(next_token_logits, dtype=tf.int32, num_samples=1), axis=1
                )
            else:
                # Greedy decoding
                next_token = tf.math.argmax(next_token_logits, axis=-1, output_type=tf.int32)

            # update generations and finished sentences
            if eos_token_ids is not None:
                # pad finished sentences if eos_token_ids exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            input_ids = tf.concat([input_ids, tf.expand_dims(tokens_to_add, -1)], 1)

            if eos_token_ids is not None:
                for eos_token_id in eos_token_ids:
                    eos_in_sents = tokens_to_add == eos_token_id
                    # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                    is_sents_unfinished_and_token_to_add_is_eos = tf.math.multiply(
                        unfinished_sents, tf.cast(eos_in_sents, tf.int32)
                    )
                    sent_lengths = (
                        sent_lengths * (1 - is_sents_unfinished_and_token_to_add_is_eos)
                        + cur_len * is_sents_unfinished_and_token_to_add_is_eos
                    )

                    # unfinished_sents is set to zero if eos in sentence
                    unfinished_sents -= is_sents_unfinished_and_token_to_add_is_eos

            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if tf.math.reduce_max(unfinished_sents) == 0:
                break

        # if there are different sentences lengths in the batch, some batches have to be padded
        min_sent_length = tf.math.reduce_min(sent_lengths)
        max_sent_length = tf.math.reduce_max(sent_lengths)
        if min_sent_length != max_sent_length:
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            padding = tf.ones([batch_size, max_sent_length.numpy()], dtype=tf.int32) * pad_token_id

            # create length masks for tf.where operation
            broad_casted_sent_lengths = tf.broadcast_to(
                tf.expand_dims(sent_lengths, -1), [batch_size, max_sent_length]
            )
            broad_casted_range = tf.transpose(
                tf.broadcast_to(tf.expand_dims(tf.range(max_length), -1), [max_length, batch_size])
            )

            decoded = tf.where(broad_casted_range < broad_casted_sent_lengths, input_ids, padding)
        else:
            decoded = input_ids

        return decoded

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
    ):
        pass


def tf_top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    logits_shape = shape_list(logits)

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits_shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < tf.math.top_k(logits, k=top_k)[0][..., -1, None]
        logits = set_tensor_by_indices_to_value(logits, indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_indices = tf.argsort(logits, direction="DESCENDING")
        sorted_logits = tf.gather(
            logits, sorted_indices, axis=-1, batch_dims=1
        )  # expects logits to be of dim (batch_size, vocab_size)

        cumulative_probs = tf.math.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p

        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove = tf.concat(
                [
                    tf.zeros_like(sorted_indices_to_remove[:, :min_tokens_to_keep]),
                    sorted_indices_to_remove[:, min_tokens_to_keep:],
                ],
                -1,
            )

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = tf.concat(
            [tf.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, 1:]], -1,
        )
        # scatter sorted tensors to original indexing
        indices_to_remove = scatter_values_on_batch_indices(sorted_indices_to_remove, sorted_indices)
        logits = set_tensor_by_indices_to_value(logits, indices_to_remove, filter_value)
    return logits


def scatter_values_on_batch_indices(values, batch_indices):
    shape = shape_list(batch_indices)
    # broadcast batch dim to shape
    broad_casted_batch_dims = tf.reshape(tf.broadcast_to(tf.expand_dims(tf.range(shape[0]), axis=-1), shape), [1, -1])
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), shape)


def set_tensor_by_indices_to_value(tensor, indices, value):
    # create value_tensor since tensor value assignment is not possible in TF
    value_tensor = tf.zeros_like(tensor) + value
    return tf.where(indices, value_tensor, tensor)


class TFConv1D(tf.keras.layers.Layer):
    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        """ TFConv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super().__init__(**kwargs)
        self.nf = nf
        self.nx = nx
        self.initializer_range = initializer_range

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight", shape=[self.nx, self.nf], initializer=get_initializer(self.initializer_range)
        )
        self.bias = self.add_weight("bias", shape=[1, self.nf], initializer=tf.zeros_initializer())

    def call(self, x):
        bz, sl = shape_list(x)[:2]

        x = tf.reshape(x, [-1, self.nx])
        x = tf.matmul(x, self.weight) + self.bias

        x = tf.reshape(x, [bz, sl, self.nf])

        return x


class TFSharedEmbeddings(tf.keras.layers.Layer):
    """Construct shared token embeddings.
    """

    def __init__(self, vocab_size, hidden_size, initializer_range=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = hidden_size ** -0.5 if initializer_range is None else initializer_range

    def build(self, input_shape):
        """Build shared word embedding layer
        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        self.weight = self.add_weight(
            "weight", shape=[self.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range)
        )
        super().build(input_shape)

    def call(self, inputs, mode="embedding"):
        """Get token embeddings of inputs.
        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
                shape [batch_size, length, embedding_size]; (2) mode == "linear", output
                linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, input_ids):
        """Applies embedding based on inputs tensor."""
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [..., hidden_size]
            Returns:
                float32 tensor with shape [..., vocab_size].
        """
        first_dims = shape_list(inputs)[:-1]

        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])


class TFSequenceSummary(tf.keras.layers.Layer):
    r""" Compute a single vector summary of a sequence hidden states according to various possibilities:
        Args of the config class:
            summary_type:
                - 'last' => [default] take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj: Add a projection after the vector extraction
            summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_activation: 'tanh' => add a tanh activation to the output, Other => no activation. Default
            summary_first_dropout: Add a dropout before the projection and activation
            summary_last_dropout: Add a dropout after the projection and activation
    """

    def __init__(self, config, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)

        self.summary_type = config.summary_type if hasattr(config, "summary_use_proj") else "last"
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.has_summary = hasattr(config, "summary_use_proj") and config.summary_use_proj
        if self.has_summary:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = tf.keras.layers.Dense(
                num_classes, kernel_initializer=get_initializer(initializer_range), name="summary"
            )

        self.has_activation = hasattr(config, "summary_activation") and config.summary_activation == "tanh"
        if self.has_activation:
            self.activation = tf.keras.activations.tanh

        self.has_first_dropout = hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0
        if self.has_first_dropout:
            self.first_dropout = tf.keras.layers.Dropout(config.summary_first_dropout)

        self.has_last_dropout = hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0
        if self.has_last_dropout:
            self.last_dropout = tf.keras.layers.Dropout(config.summary_last_dropout)

    def call(self, inputs, training=False):
        """ hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        if not isinstance(inputs, (dict, tuple, list)):
            hidden_states = inputs
            cls_index = None
        elif isinstance(inputs, (tuple, list)):
            hidden_states = inputs[0]
            cls_index = inputs[1] if len(inputs) > 1 else None
            assert len(inputs) <= 2, "Too many inputs."
        else:
            hidden_states = inputs.get("hidden_states")
            cls_index = inputs.get("cls_index", None)

        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = tf.reduce_mean(hidden_states, axis=1)
        elif self.summary_type == "cls_index":
            hidden_shape = shape_list(hidden_states)  # e.g. [batch, num choices, seq length, hidden dims]
            if cls_index is None:
                cls_index = tf.fill(
                    hidden_shape[:-2], hidden_shape[-2] - 1
                )  # A tensor full of shape [batch] or [batch, num choices] full of sequence length
            cls_shape = shape_list(cls_index)
            if len(cls_shape) <= len(hidden_shape) - 2:
                cls_index = cls_index[..., tf.newaxis]
            # else:
            # cls_index = cls_index[..., tf.newaxis]
            # cls_index = cls_index.expand((-1,) * (cls_index.dim()-1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = tf.gather(hidden_states, cls_index, batch_dims=len(hidden_shape) - 2)
            output = tf.squeeze(
                output, axis=len(hidden_shape) - 2
            )  # shape of output: (batch, num choices, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        if self.has_first_dropout:
            output = self.first_dropout(output, training=training)

        if self.has_summary:
            output = self.summary(output)

        if self.has_activation:
            output = self.activation(output)

        if self.has_last_dropout:
            output = self.last_dropout(output, training=training)

        return output


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
    Args:
        initializer_range: float, initializer range for stddev.
    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
