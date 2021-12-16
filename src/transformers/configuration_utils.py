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
import os
import re
import warnings
from typing import Any, Dict, Optional, Tuple, Union

from packaging import version

from . import __version__
from .file_utils import (
    CONFIG_NAME,
    PushToHubMixin,
    cached_path,
    copy_func,
    get_list_of_files,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    is_torch_available,
)
from .utils import logging


logger = logging.get_logger(__name__)
FULL_CONFIGURATION_FILE = "config.json"
_re_configuration_file = re.compile(r"config\.(.*)\.json")


class PretrainedConfig(PushToHubMixin):
    r"""
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    <Tip>

    A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
    initialize a model does **not** load the model weights. It only affects the model's configuration.

    </Tip>

    Class attributes (overridden by derived classes):

    - **model_type** (`str`) -- An identifier for the model type, serialized into the JSON file, and used to recreate
      the correct object in [`~transformers.AutoConfig`].
    - **is_composition** (`bool`) -- Whether the config class is composed of multiple sub-configs. In this case the
      config has to be initialized from two or more configs of type [`~transformers.PretrainedConfig`] like:
      [`~transformers.EncoderDecoderConfig`] or [`~RagConfig`].
    - **keys_to_ignore_at_inference** (`List[str]`) -- A list of keys to ignore by default when looking at dictionary
      outputs of the model during inference.
    - **attribute_map** (`Dict[str, str]`) -- A dict that maps model specific attribute names to the standardized
      naming of attributes.

    Common attributes (present in all subclasses):

    - **vocab_size** (`int`) -- The number of tokens in the vocabulary, which is also the first dimension of the
      embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
    - **hidden_size** (`int`) -- The hidden size of the model.
    - **num_attention_heads** (`int`) -- The number of attention heads used in the multi-head attention layers of the
      model.
    - **num_hidden_layers** (`int`) -- The number of blocks in the model.

    Arg:
        name_or_path (`str`, _optional_, defaults to `""`):
            Store the string that was passed to [`PreTrainedModel.from_pretrained`] or
            [`TFPreTrainedModel.from_pretrained`] as `pretrained_model_name_or_path` if the configuration was created
            with such a method.
        output_hidden_states (`bool`, _optional_, defaults to `False`):
            Whether or not the model should return all hidden-states.
        output_attentions (`bool`, _optional_, defaults to `False`):
            Whether or not the model should returns all attentions.
        return_dict (`bool`, _optional_, defaults to `True`):
            Whether or not the model should return a [`~transformers.file_utils.ModelOutput`] instead of a plain tuple.
        is_encoder_decoder (`bool`, _optional_, defaults to `False`):
            Whether the model is used as an encoder/decoder or not.
        is_decoder (`bool`, _optional_, defaults to `False`):
            Whether the model is used as decoder or not (in which case it's used as an encoder).
        cross_attention_hidden_size** (`bool`, _optional_):
            The hidden size of the cross-attention layer in case the model is used as a decoder in an encoder-decoder
            setting and the cross-attention hidden dimension differs from `self.config.hidden_size`.
        add_cross_attention (`bool`, _optional_, defaults to `False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the [`EncoderDecoderModel`] class, which consists of all models
            in `AUTO_MODELS_FOR_CAUSAL_LM`.
        tie_encoder_decoder (`bool`, _optional_, defaults to `False`):
            Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder
            and decoder model to have the exact same parameter names.
        prune_heads (`Dict[int, List[int]]`, _optional_, defaults to `{}`):
            Pruned heads of the model. The keys are the selected layer indices and the associated values, the list of
            heads to prune in said layer.

            For instance `{1: [0, 2], 2: [2, 3]}` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        chunk_size_feed_forward (`int`, _optional_, defaults to `0`):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means that
            the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes `n` <
            sequence_length embeddings at a time. For more information on feed forward chunking, see [How does Feed
            Forward Chunking work?](../glossary.html#feed-forward-chunking).

        > Parameters for sequence generation

        max_length (`int`, _optional_, defaults to 20):
            Maximum length that will be used by default in the `generate` method of the model.
        min_length (`int`, _optional_, defaults to 10):
            Minimum length that will be used by default in the `generate` method of the model.
        do_sample (`bool`, _optional_, defaults to `False`):
            Flag that will be used by default in the `generate` method of the model. Whether or not to use sampling ;
            use greedy decoding otherwise.
        early_stopping (`bool`, _optional_, defaults to `False`):
            Flag that will be used by default in the `generate` method of the model. Whether to stop the beam search
            when at least `num_beams` sentences are finished per batch or not.
        num_beams (`int`, _optional_, defaults to 1):
            Number of beams for beam search that will be used by default in the `generate` method of the model. 1 means
            no beam search.
        num_beam_groups (`int`, _optional_, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams
            that will be used by default in the `generate` method of the model. 1 means no group beam search.
        diversity_penalty (`float`, _optional_, defaults to 0.0):
            Value to control diversity for group beam search. that will be used by default in the `generate` method of
            the model. 0 means no diversity penalty. The higher the penalty, the more diverse are the outputs.
        temperature (`float`, _optional_, defaults to 1):
            The value used to module the next token probabilities that will be used by default in the `generate` method
            of the model. Must be strictly positive.
        top_k (`int`, _optional_, defaults to 50):
            Number of highest probability vocabulary tokens to keep for top-k-filtering that will be used by default in
            the `generate` method of the model.
        top_p (`float`, _optional_, defaults to 1):
            Value that will be used by default in the `generate` method of the model for `top_p`. If set to float < 1,
            only the most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
        repetition_penalty (`float`, _optional_, defaults to 1):
            Parameter for repetition penalty that will be used by default in the `generate` method of the model. 1.0
            means no penalty.
        length_penalty (`float`, _optional_, defaults to 1):
            Exponential penalty to the length that will be used by default in the `generate` method of the model.
        no_repeat_ngram_size (`int`, _optional_, defaults to 0) -- Value that will be used by default in the
            `generate` method of the model for `no_repeat_ngram_size`. If set to int > 0, all ngrams of that size can
            only occur once.
        encoder_no_repeat_ngram_size (`int`, _optional_, defaults to 0) -- Value that will be used by
            default in the `generate` method of the model for `encoder_no_repeat_ngram_size`. If set to int > 0, all
            ngrams of that size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`.
        bad_words_ids (`List[int]`, _optional_):
            List of token ids that are not allowed to be generated that will be used by default in the `generate`
            method of the model. In order to get the tokens of the words that should not appear in the generated text,
            use `tokenizer.encode(bad_word, add_prefix_space=True)`.
        num_return_sequences (`int`, _optional_, defaults to 1):
            Number of independently computed returned sequences for each element in the batch that will be used by
            default in the `generate` method of the model.
        output_scores (`bool`, _optional_, defaults to `False`):
            Whether the model should return the logits when used for generation.
        return_dict_in_generate (`bool`, _optional_, defaults to `False`):
            Whether the model should return a [`~transformers.file_utils.ModelOutput`] instead of a `torch.LongTensor`.
        forced_bos_token_id (`int`, _optional_):
            The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for
            multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be the target
            language token.
        forced_eos_token_id (`int`, _optional_):
            The id of the token to force as the last generated token when `max_length` is reached.
        remove_invalid_values (`bool`, _optional_):
            Whether to remove possible _nan_ and _inf_ outputs of the model to prevent the generation method to crash.
            Note that using `remove_invalid_values` can slow down generation.

        > Parameters for fine-tuning tasks

        architectures (`List[str]`, _optional_): Model architectures that can be used with the model pretrained weights.
        finetuning_task (`str`, _optional_):
            Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow
            or PyTorch) checkpoint.
        id2label (`Dict[int, str]`, _optional_):
            A map from index (for instance prediction index, or target index) to label.
        label2id (`Dict[str, int]`, _optional_): A map from label to index for the model.
        num_labels (`int`, _optional_):
            Number of labels to use in the last layer added to the model, typically for a classification task.
        task_specific_params (`Dict[str, Any]`, _optional_):
            Additional keyword arguments to store for the current task.
        problem_type (`str`, _optional_):
            Problem type for `XxxForSequenceClassification` models. Can be one of `"regression"`,
            `"single_label_classification"` or `"multi_label_classification"`.

        > Parameters linked to the tokenizer

        tokenizer_class (`str`, _optional_):
            The name of the associated tokenizer class to use (if none is set, will use the tokenizer associated to the
            model by default).
        prefix (`str`, _optional_):
            A specific prompt that should be added at the beginning of each text before calling the model.
        bos_token_id (`int`, _optional_): The id of the _beginning-of-stream_ token.
        pad_token_id (`int`, _optional_): The id of the _padding_ token.
        eos_token_id (`int`, _optional_): The id of the _end-of-stream_ token.
        decoder_start_token_id (`int`, _optional_):
            If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token.
        sep_token_id (`int`, _optional_): The id of the _separation_ token.

        > PyTorch specific parameters

        torchscript (`bool`, _optional_, defaults to `False`):
            Whether or not the model should be used with Torchscript.
        tie_word_embeddings (`bool`, _optional_, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        torch_dtype (`str`, _optional_):
            The `dtype` of the weights. This attribute can be used to initialize the model to a non-default `dtype`
            (which is normally `float32`) and thus allow for optimal storage allocation. For example, if the saved
            model is `float16`, ideally we want to load it back using the minimal amount of memory needed to load
            `float16` weights. Since the config object is stored in plain text, this attribute contains just the
            floating type string without the `torch.` prefix. For example, for `torch.float16` ``torch_dtype` is the
            `"float16"` string.

            This attribute is currently not being used during model loading time, but this may change in the future
            versions. But we can already start preparing for the future by saving the dtype with save_pretrained.

        > TensorFlow specific parameters

        use_bfloat16 (`bool`, _optional_, defaults to `False`):
            Whether or not the model should use BFloat16 scalars (only used by some TensorFlow models).
    """
    model_type: str = ""
    is_composition: bool = False
    attribute_map: Dict[str, str] = {}

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(self, **kwargs):
        # Attributes with defaults
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
        self.torch_dtype = kwargs.pop("torch_dtype", None)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.tie_word_embeddings = kwargs.pop(
            "tie_word_embeddings", True
        )  # Whether input and output word embeddings should be tied for all MLM, LM and Seq2Seq models.

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.cross_attention_hidden_size = kwargs.pop("cross_attention_hidden_size", None)
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.min_length = kwargs.pop("min_length", 0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        self.output_scores = kwargs.pop("output_scores", False)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.id2label = kwargs.pop("id2label", None)
        self.label2id = kwargs.pop("label2id", None)
        if self.id2label is not None:
            kwargs.pop("num_labels", None)
            self.id2label = dict((int(key), value) for key, value in self.id2label.items())
            # Keys are always strings in JSON so convert ids to int here.
        else:
            self.num_labels = kwargs.pop("num_labels", 2)

        if self.torch_dtype is not None and isinstance(self.torch_dtype, str):
            # we will start using self.torch_dtype in v5, but to be consistent with
            # from_pretrained's torch_dtype arg convert it to an actual torch.dtype object
            if is_torch_available():
                import torch

                self.torch_dtype = getattr(torch, self.torch_dtype)

        # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)

        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # task specific arguments
        self.task_specific_params = kwargs.pop("task_specific_params", None)

        # regression / multi-label classification
        self.problem_type = kwargs.pop("problem_type", None)
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {self.problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )

        # TPU arguments
        if kwargs.pop("xla_device", None) is not None:
            logger.warning(
                "The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can "
                "safely remove it from your `config.json` file."
            )

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))

        # Drop the transformers version info
        self.transformers_version = kwargs.pop("transformers_version", None)

        # Deal with gradient checkpointing
        if kwargs.get("gradient_checkpointing", False):
            warnings.warn(
                "Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 "
                "Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the "
                "`Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`."
            )

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    @property
    def name_or_path(self) -> str:
        return self._name_or_path

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    @property
    def use_return_dict(self) -> bool:
        """
        :obj:`bool`: Whether or not return :class:`~transformers.file_utils.ModelOutput` instead of tuples.
        """
        # If torchscript is set, force `return_dict=False` to avoid jit errors
        return self.return_dict and not self.torchscript

    @property
    def num_labels(self) -> int:
        """
        :obj:`int`: The number of labels for classification models.
        """
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels: int):
        if not hasattr(self, "id2label") or self.id2label is None or len(self.id2label) != num_labels:
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                .. warning::

                    Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
                    :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
                    pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
                    instead.

            kwargs:
                Additional key word arguments passed along to the
                :meth:`~transformers.file_utils.PushToHubMixin.push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Configuration pushed to the hub in this commit: {url}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        r"""
        Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pretrained model
        configuration.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained model configuration hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a configuration file saved using the
                  :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g., ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`, e.g.,
                  ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final configuration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e.,
                the part of ``kwargs`` which has not been used to update ``config`` and is otherwise ignored.
            kwargs (:obj:`Dict[str, Any]`, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the ``return_unused_kwargs`` keyword parameter.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.


        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from this pretrained model.

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from huggingface.co and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False)
            assert config.output_attentions == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attentions == True
            assert unused_kwargs == {'foo': False}

        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warn(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        :class:`~transformers.PretrainedConfig` using ``from_dict``.



        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            configuration_file = get_configuration_file(
                pretrained_model_name_or_path,
                revision=revision,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )

            if os.path.isdir(pretrained_model_name_or_path):
                config_file = os.path.join(pretrained_model_name_or_path, configuration_file)
            else:
                config_file = hf_bucket_url(
                    pretrained_model_name_or_path, filename=configuration_file, revision=revision, mirror=None
                )

        try:
            # Load from URL or cache if already cached
            resolved_config_file = cached_path(
                config_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
            )
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)

        except EnvironmentError as err:
            logger.error(err)
            msg = (
                f"Can't load config for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n"
                f"  (make sure '{pretrained_model_name_or_path}' is not a path to a local directory with something else, in that case)\n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {CONFIG_NAME} file\n\n"
            )

            if revision is not None:
                msg += f"- or '{revision}' is a valid git identifier (branch name, a tag name, or a commit id) that exists for this model name as listed on its model page on 'https://huggingface.co/models'\n\n"

            raise EnvironmentError(msg)

        except (json.JSONDecodeError, UnicodeDecodeError):
            msg = (
                f"Couldn't reach server at '{config_file}' to download configuration file or "
                "configuration file is not a valid JSON file. "
                f"Please check network or file content here: {resolved_config_file}."
            )
            raise EnvironmentError(msg)

        if resolved_config_file == config_file:
            logger.info(f"loading configuration file {config_file}")
        else:
            logger.info(f"loading configuration file {config_file} from cache at {resolved_config_file}")

        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.PretrainedConfig.get_config_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from those parameters.
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
                if key != "torch_dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Model config {config}")
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PretrainedConfig":
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from the path to a JSON file of parameters.

        Args:
            json_file (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                key not in default_config_dict
                or key == "transformers_version"
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        self.dict_torch_dtype_to_str(serializable_config_dict)

        return serializable_config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Transformers version when serializing the model
        output["transformers_version"] = __version__

        self.dict_torch_dtype_to_str(output)

        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: Dict[str, Any]):
        """
        Updates attributes of this class with attributes from ``config_dict``.

        Args:
            config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from ``update_str``.

        The expected format is ints, floats and strings as is, and for booleans use ``true`` or ``false``. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

        The keys to change have to already exist in the config object.

        Args:
            update_str (:obj:`str`): String with attributes that should be updated for this class.

        """

        d = dict(x.split("=") for x in update_str.split(","))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")

            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise ValueError(
                    f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
                )

            setattr(self, k, v)

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary has a `torch_dtype` key and if it's not None, converts torch.dtype to a
        string of just the type. For example, :obj:`torch.float32` get converted into `"float32"` string, which can
        then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]


def get_configuration_file(
    path_or_repo: Union[str, os.PathLike],
    revision: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    local_files_only: bool = False,
) -> str:
    """
    Get the configuration file to use for this version of transformers.

    Args:
        path_or_repo (:obj:`str` or :obj:`os.PathLike`):
            Can be either the id of a repo on huggingface.co or a path to a `directory`.
        revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
            identifier allowed by git.
        use_auth_token (:obj:`str` or `bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
        local_files_only (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to only rely on local files and not to attempt to download any files.

    Returns:
        :obj:`str`: The configuration file to use.
    """
    # Inspect all files from the repo/folder.
    all_files = get_list_of_files(
        path_or_repo, revision=revision, use_auth_token=use_auth_token, local_files_only=local_files_only
    )
    configuration_files_map = {}
    for file_name in all_files:
        search = _re_configuration_file.search(file_name)
        if search is not None:
            v = search.groups()[0]
            configuration_files_map[v] = file_name
    available_versions = sorted(configuration_files_map.keys())

    # Defaults to FULL_CONFIGURATION_FILE and then try to look at some newer versions.
    configuration_file = FULL_CONFIGURATION_FILE
    transformers_version = version.parse(__version__)
    for v in available_versions:
        if version.parse(v) <= transformers_version:
            configuration_file = configuration_files_map[v]
        else:
            # No point going further since the versions are sorted.
            break

    return configuration_file


PretrainedConfig.push_to_hub = copy_func(PretrainedConfig.push_to_hub)
PretrainedConfig.push_to_hub.__doc__ = PretrainedConfig.push_to_hub.__doc__.format(
    object="config", object_class="AutoConfig", object_files="configuration file"
)
