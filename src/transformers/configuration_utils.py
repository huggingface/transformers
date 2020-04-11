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
""" Configuration base class and utilities."""


import copy
import json
import logging
import os
from typing import Dict, Optional, Tuple

from .file_utils import CONFIG_NAME, cached_path, hf_bucket_url, is_remote_url


logger = logging.getLogger(__name__)


class PretrainedConfig(object):
    r""" Base class for all configuration classes.
        Handles a few parameters common to all models' configurations as well as methods for loading/downloading/saving configurations.

        Note:
            A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to initialize a model does **not** load the model weights.
            It only affects the model's configuration.

        Class attributes (overridden by derived classes):
            - ``pretrained_config_archive_map``: a python ``dict`` with `shortcut names` (string) as keys and `url` (string) of associated pretrained model configurations as values.
            - ``model_type``: a string that identifies the model type, that we serialize into the JSON file, and that we use to recreate the correct object in :class:`~transformers.AutoConfig`.

        Args:
            finetuning_task (:obj:`string` or :obj:`None`, `optional`, defaults to :obj:`None`):
                Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow or PyTorch) checkpoint.
            num_labels (:obj:`int`, `optional`, defaults to `2`):
                Number of classes to use when the model is a classification model (sequences/tokens)
            output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Should the model returns attentions weights.
            output_hidden_states (:obj:`string`, `optional`, defaults to :obj:`False`):
                Should the model returns all hidden-states.
            torchscript (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Is the model used with Torchscript (for PyTorch models).
    """
    pretrained_config_archive_map = {}  # type: Dict[str, str]
    model_type = ""  # type: str

    def __init__(self, **kwargs):
        # Attributes with defaults
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_past = kwargs.pop("output_past", True)  # Not used by all models
        self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.min_length = kwargs.pop("min_length", 0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.id2label = kwargs.pop("id2label", {i: "LABEL_{}".format(i) for i in range(self.num_labels)})
        self.id2label = dict((int(key), value) for key, value in self.id2label.items())
        self.label2id = kwargs.pop("label2id", dict(zip(self.id2label.values(), self.id2label.keys())))
        self.label2id = dict((key, int(value)) for key, value in self.label2id.items())

        # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # task specific arguments
        self.task_specific_params = kwargs.pop("task_specific_params", None)

        # TPU arguments
        self.xla_device = kwargs.pop("xla_device", None)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @property
    def num_labels(self):
        return self._num_labels

    @num_labels.setter
    def num_labels(self, num_labels):
        self._num_labels = num_labels
        self.id2label = {i: "LABEL_{}".format(i) for i in range(self.num_labels)}
        self.id2label = dict((int(key), value) for key, value in self.id2label.items())
        self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))
        self.label2id = dict((key, int(value)) for key, value in self.label2id.items())

    def save_pretrained(self, save_directory):
        """
        Save a configuration object to the directory `save_directory`, so that it
        can be re-loaded using the :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`string`):
                Directory where the configuration JSON file will be saved.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PretrainedConfig":
        r"""

        Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pre-trained model configuration.

        Args:
            pretrained_model_name_or_path (:obj:`string`):
                either:
                  - a string with the `shortcut name` of a pre-trained model configuration to load from cache or
                    download, e.g.: ``bert-base-uncased``.
                  - a string with the `identifier name` of a pre-trained model configuration that was user-uploaded to
                    our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                  - a path to a `directory` containing a configuration file saved using the
                    :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                  - a path or url to a saved configuration JSON `file`, e.g.:
                    ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`string`, `optional`):
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            kwargs (:obj:`Dict[str, any]`, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is
                controlled by the `return_unused_kwargs` keyword parameter.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Force to (re-)download the model weights and configuration files and override the cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.
            proxies (:obj:`Dict`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g.:
                :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.`
                The proxies are used on each request.
            return_unused_kwargs: (`optional`) bool:
                If False, then this function returns just the final configuration object.
                If True, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs` is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part
                of kwargs which has not been used to update `config` and is otherwise ignored.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str, pretrained_config_archive_map: Optional[Dict] = None, **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used
        for instantiating a Config using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (:obj:`string`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            pretrained_config_archive_map: (:obj:`Dict[str, str]`, `optional`) Dict:
                A map of `shortcut names` to `url`. By default, will use the current class attribute.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary that will be used to instantiate the configuration object.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if pretrained_config_archive_map is None:
            pretrained_config_archive_map = cls.pretrained_config_archive_map

        if pretrained_model_name_or_path in pretrained_config_archive_map:
            config_file = pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            config_file = hf_bucket_url(pretrained_model_name_or_path, postfix=CONFIG_NAME)

        try:
            # Load from URL or cache if already cached
            resolved_config_file = cached_path(
                config_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
            )
            # Load config dict
            if resolved_config_file is None:
                raise EnvironmentError
            config_dict = cls._dict_from_json_file(resolved_config_file)

        except EnvironmentError:
            if pretrained_model_name_or_path in pretrained_config_archive_map:
                msg = "Couldn't reach server at '{}' to download pretrained model configuration file.".format(
                    config_file
                )
            else:
                msg = (
                    "Can't load '{}'. Make sure that:\n\n"
                    "- '{}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    "- or '{}' is the correct path to a directory containing a '{}' file\n\n".format(
                        pretrained_model_name_or_path,
                        pretrained_model_name_or_path,
                        pretrained_model_name_or_path,
                        CONFIG_NAME,
                    )
                )
            raise EnvironmentError(msg)

        except json.JSONDecodeError:
            msg = (
                "Couldn't reach server at '{}' to download configuration file or "
                "configuration file is not a valid JSON file. "
                "Please check network or file content here: {}.".format(config_file, resolved_config_file)
            )
            raise EnvironmentError(msg)

        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading configuration file {} from cache at {}".format(config_file, resolved_config_file))

        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "PretrainedConfig":
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
            kwargs (:obj:`Dict[str, any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: str) -> "PretrainedConfig":
        """
        Constructs a `Config` from the path to a json file of parameters.

        Args:
            json_file (:obj:`string`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return "{} {}".format(self.__class__.__name__, self.to_json_string())

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.

        Returns:
            :obj:`string`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """
        Save this instance to a json file.

        Args:
            json_file_path (:obj:`string`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def update(self, config_dict: Dict):
        """
        Updates attributes of this class
        with attributes from `config_dict`.

        Args:
            :obj:`Dict[str, any]`: Dictionary of attributes that shall be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)
