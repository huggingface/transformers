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
"""Configuration base class and utilities."""

import copy
import json
import os
import warnings
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

from packaging import version

from . import __version__
from .dynamic_module_utils import custom_object_save
from .modeling_gguf_pytorch_utils import load_gguf_checkpoint
from .utils import (
    CONFIG_NAME,
    PushToHubMixin,
    cached_file,
    copy_func,
    download_url,
    extract_commit_hash,
    is_remote_url,
    is_torch_available,
    logging,
)
from .utils.generic import is_timm_config_dict


if TYPE_CHECKING:
    import torch


logger = logging.get_logger(__name__)


# type hinting: specifying the type of config class that inherits from PretrainedConfig
SpecificPretrainedConfigType = TypeVar("SpecificPretrainedConfigType", bound="PretrainedConfig")


class PretrainedConfig(PushToHubMixin):
    # no-format
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
    - **has_no_defaults_at_init** (`bool`) -- Whether the config class can be initialized without providing input arguments.
      Some configurations requires inputs to be defined at init and have no default values, usually these are composite configs,
      (but not necessarily) such as [`~transformers.EncoderDecoderConfig`] or [`~RagConfig`]. They have to be initialized from
      two or more configs of type [`~transformers.PretrainedConfig`].
    - **keys_to_ignore_at_inference** (`list[str]`) -- A list of keys to ignore by default when looking at dictionary
      outputs of the model during inference.
    - **attribute_map** (`dict[str, str]`) -- A dict that maps model specific attribute names to the standardized
      naming of attributes.
    - **base_model_tp_plan** (`dict[str, Any]`) -- A dict that maps sub-modules FQNs of a base model to a tensor
      parallel plan applied to the sub-module when `model.tensor_parallel` is called.
    - **base_model_pp_plan** (`dict[str, tuple[list[str]]]`) -- A dict that maps child-modules of a base model to a
      pipeline parallel plan that enables users to place the child-module on the appropriate device.

    Common attributes (present in all subclasses):

    - **vocab_size** (`int`) -- The number of tokens in the vocabulary, which is also the first dimension of the
      embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
    - **hidden_size** (`int`) -- The hidden size of the model.
    - **num_attention_heads** (`int`) -- The number of attention heads used in the multi-head attention layers of the
      model.
    - **num_hidden_layers** (`int`) -- The number of blocks in the model.

    <Tip warning={true}>

    Setting parameters for sequence generation in the model config is deprecated. For backward compatibility, loading
    some of them will still be possible, but attempting to overwrite them will throw an exception -- you should set
    them in a [~transformers.GenerationConfig]. Check the documentation of [~transformers.GenerationConfig] for more
    information about the individual parameters.

    </Tip>

    Arg:
        name_or_path (`str`, *optional*, defaults to `""`):
            Store the string that was passed to [`PreTrainedModel.from_pretrained`] or
            [`TFPreTrainedModel.from_pretrained`] as `pretrained_model_name_or_path` if the configuration was created
            with such a method.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return all hidden-states.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not the model should returns all attentions.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return a [`~transformers.utils.ModelOutput`] instead of a plain tuple.
        is_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as an encoder/decoder or not.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether to only use the decoder in an encoder-decoder architecture, otherwise it has no effect on
            decoder-only or encoder-only architectures.
        cross_attention_hidden_size (`bool`, *optional*):
            The hidden size of the cross-attention layer in case the model is used as a decoder in an encoder-decoder
            setting and the cross-attention hidden dimension differs from `self.config.hidden_size`.
        add_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the [`EncoderDecoderModel`] class, which consists of all models
            in `AUTO_MODELS_FOR_CAUSAL_LM`.
        tie_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder
            and decoder model to have the exact same parameter names.
        prune_heads (`dict[int, list[int]]`, *optional*, defaults to `{}`):
            Pruned heads of the model. The keys are the selected layer indices and the associated values, the list of
            heads to prune in said layer.

            For instance `{1: [0, 2], 2: [2, 3]}` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        chunk_size_feed_forward (`int`, *optional*, defaults to `0`):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means that
            the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes `n` <
            sequence_length embeddings at a time. For more information on feed forward chunking, see [How does Feed
            Forward Chunking work?](../glossary.html#feed-forward-chunking).

        > Parameters for fine-tuning tasks

        architectures (`list[str]`, *optional*):
            Model architectures that can be used with the model pretrained weights.
        finetuning_task (`str`, *optional*):
            Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow
            or PyTorch) checkpoint.
        id2label (`dict[int, str]`, *optional*):
            A map from index (for instance prediction index, or target index) to label.
        label2id (`dict[str, int]`, *optional*):
            A map from label to index for the model.
        num_labels (`int`, *optional*):
            Number of labels to use in the last layer added to the model, typically for a classification task.
        task_specific_params (`dict[str, Any]`, *optional*):
            Additional keyword arguments to store for the current task.
        problem_type (`str`, *optional*):
            Problem type for `XxxForSequenceClassification` models. Can be one of `"regression"`,
            `"single_label_classification"` or `"multi_label_classification"`.

        > Parameters linked to the tokenizer

        tokenizer_class (`str`, *optional*):
            The name of the associated tokenizer class to use (if none is set, will use the tokenizer associated to the
            model by default).
        prefix (`str`, *optional*):
            A specific prompt that should be added at the beginning of each text before calling the model.
        bos_token_id (`int`, *optional*):
            The id of the _beginning-of-stream_ token.
        pad_token_id (`int`, *optional*):
            The id of the _padding_ token.
        eos_token_id (`int`, *optional*):
            The id of the _end-of-stream_ token.
        decoder_start_token_id (`int`, *optional*):
            If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token.
        sep_token_id (`int`, *optional*):
            The id of the _separation_ token.

        > PyTorch specific parameters

        torchscript (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be used with Torchscript.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        dtype (`str`, *optional*):
            The `dtype` of the weights. This attribute can be used to initialize the model to a non-default `dtype`
            (which is normally `float32`) and thus allow for optimal storage allocation. For example, if the saved
            model is `float16`, ideally we want to load it back using the minimal amount of memory needed to load
            `float16` weights.
    """

    model_type: str = ""
    base_config_key: str = ""
    sub_configs: dict[str, type["PretrainedConfig"]] = {}
    has_no_defaults_at_init: bool = False
    attribute_map: dict[str, str] = {}
    base_model_tp_plan: Optional[dict[str, Any]] = None
    base_model_pp_plan: Optional[dict[str, tuple[list[str]]]] = None
    _auto_class: Optional[str] = None

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(
        self,
        *,
        # All models common arguments
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        torchscript: bool = False,
        dtype: Optional[Union[str, "torch.dtype"]] = None,
        # Common arguments
        pruned_heads: Optional[dict[int, list[int]]] = None,
        tie_word_embeddings: bool = True,
        chunk_size_feed_forward: int = 0,
        is_encoder_decoder: bool = False,
        is_decoder: bool = False,
        cross_attention_hidden_size: Optional[int] = None,
        add_cross_attention: bool = False,
        tie_encoder_decoder: bool = False,
        # Fine-tuning task arguments
        architectures: Optional[list[str]] = None,
        finetuning_task: Optional[str] = None,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        num_labels: Optional[int] = None,
        task_specific_params: Optional[dict[str, Any]] = None,
        problem_type: Optional[str] = None,
        # Tokenizer kwargs
        tokenizer_class: Optional[str] = None,
        prefix: Optional[str] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        sep_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        **kwargs,
    ):
        # Validation for some arguments
        if label2id is not None and not isinstance(label2id, dict):
            raise ValueError("Argument label2id should be a dictionary.")
        if id2label is not None and not isinstance(id2label, dict):
            raise ValueError("Argument id2label should be a dictionary.")
        if num_labels is not None and id2label is not None and len(id2label) != num_labels:
            logger.warning(
                f"You passed `num_labels={num_labels}` which is incompatible to "
                f"the `id2label` map of length `{len(id2label)}`."
            )
        if problem_type is not None and problem_type not in (
            "regression",
            "single_label_classification",
            "multi_label_classification",
        ):
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )
        # BC for the `torch_dtype` argument instead of the simpler `dtype`
        # Do not warn, as it would otherwise always be triggered since most configs on the hub have `torch_dtype`
        if (torch_dtype := kwargs.pop("torch_dtype", None)) is not None:
            # If both are provided, keep `dtype`
            dtype = dtype if dtype is not None else torch_dtype
        if dtype is not None and isinstance(dtype, str) and is_torch_available():
            # we will start using self.dtype in v5, but to be consistent with
            # from_pretrained's dtype arg convert it to an actual torch.dtype object
            import torch

            dtype = getattr(torch, dtype)

        # Attributes common for all models
        self.return_dict = return_dict
        self.output_hidden_states = output_hidden_states
        self.torchscript = torchscript
        self.dtype = dtype
        self._output_attentions = output_attentions  # has public property

        # Less common kwargs, only used by some models
        self.pruned_heads = pruned_heads if pruned_heads is not None else {}
        self.tie_word_embeddings = tie_word_embeddings
        self.chunk_size_feed_forward = chunk_size_feed_forward

        # Encoder-decoder models attributes
        self.is_encoder_decoder = is_encoder_decoder
        self.is_decoder = is_decoder  # used in encoder-decoder models to differentiate encoder from decoder
        self.cross_attention_hidden_size = cross_attention_hidden_size
        self.add_cross_attention = add_cross_attention
        self.tie_encoder_decoder = tie_encoder_decoder

        # Fine-tuning task attributes
        self.architectures = architectures
        self.finetuning_task = finetuning_task
        self.id2label = id2label
        self.label2id = label2id
        self.task_specific_params = task_specific_params
        self.problem_type = problem_type

        if self.id2label is None:
            self._create_id_label_maps(num_labels if num_labels is not None else 2)
        else:
            # Keys are always strings in JSON so convert ids to int here.
            self.id2label = {int(key): value for key, value in self.id2label.items()}

        # Tokenizer attributes
        self.tokenizer_class = tokenizer_class
        self.prefix = prefix
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.decoder_start_token_id = decoder_start_token_id

        # Retrocompatibility: Parameters for sequence generation. While we will keep the ability to load these
        # parameters, saving them will be deprecated. In a distant future, we won't need to load them.
        for parameter_name, default_value in self._get_global_generation_defaults().items():
            setattr(self, parameter_name, kwargs.pop(parameter_name, default_value))

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        self._commit_hash = kwargs.pop("_commit_hash", None)

        # Attention implementation to use, if relevant (it sets it recursively on sub-configs)
        self._attn_implementation = kwargs.pop("attn_implementation", None)

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

        # TODO: remove later, deprecated arguments for TF models
        self.tf_legacy_loss = kwargs.pop("tf_legacy_loss", False)
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)

    def _create_id_label_maps(self, num_labels: int):
        self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    @property
    def name_or_path(self) -> Optional[str]:
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    @property
    def output_attentions(self):
        """
        `bool`: Whether or not the model should returns all attentions.
        """
        return self._output_attentions

    @output_attentions.setter
    def output_attentions(self, value: bool):
        # If we set `output_attentions` explictily before the attn implementation, dispatch eager
        if value and self._attn_implementation is None:
            self._attn_implementation = "eager"
        if value and self._attn_implementation != "eager":
            raise ValueError(
                "The `output_attentions` attribute is not supported when using the `attn_implementation` set to "
                f"{self._attn_implementation}. Please set it to 'eager' instead."
            )
        self._output_attentions = value

    @property
    def use_return_dict(self) -> bool:
        """
        `bool`: Whether or not return [`~utils.ModelOutput`] instead of tuples.
        """
        # If torchscript is set, force `return_dict=False` to avoid jit errors
        return self.return_dict and not self.torchscript

    @property
    def num_labels(self) -> int:
        """
        `int`: The number of labels for classification models.
        """
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels: int):
        # we do not store `num_labels` attribute in config, but instead
        # compute it based on the length of the `id2label` map
        if self.id2label is None or self.num_labels != num_labels:
            self._create_id_label_maps(num_labels)

    @property
    def _attn_implementation(self):
        return self._attn_implementation_internal

    @_attn_implementation.setter
    def _attn_implementation(self, value: Optional[Union[str, dict]]):
        """We set it recursively on the sub-configs as well"""
        # Set if for current config
        attn_implementation = value if not isinstance(value, dict) else value.get("", self._attn_implementation)
        self._attn_implementation_internal = attn_implementation

        # Set it recursively on the subconfigs
        for subconfig_key in self.sub_configs:
            subconfig = getattr(self, subconfig_key, None)
            if subconfig is not None:
                sub_implementation = (
                    value if not isinstance(value, dict) else value.get(subconfig_key, subconfig._attn_implementation)
                )
                subconfig._attn_implementation = sub_implementation

    @property
    def torch_dtype(self):
        logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
        return self.dtype

    @torch_dtype.setter
    def torch_dtype(self, value):
        logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
        self.dtype = value

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self._set_token_in_kwargs(kwargs)

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        non_default_generation_parameters = self._get_non_default_generation_parameters()
        if len(non_default_generation_parameters) > 0:
            # TODO (joao): this should be an exception if the user has modified the loaded config. See #33886
            warnings.warn(
                "Some non-default generation parameters are set in the model config. These should go into either a) "
                "`model.generation_config` (as opposed to `model.config`); OR b) a GenerationConfig file "
                "(https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model)."
                "This warning will become an exception in the future."
                f"\nNon-default generation parameters: {str(non_default_generation_parameters)}",
                UserWarning,
            )

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # This attribute is important to know on load, but should not be serialized on save.
        if "transformers_weights" in self:
            delattr(self, "transformers_weights")

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    @staticmethod
    def _set_token_in_kwargs(kwargs, token=None):
        """Temporary method to deal with `token` and `use_auth_token`.

        This method is to avoid apply the same changes in all model config classes that overwrite `from_pretrained`.

        Need to clean up `use_auth_token` in a follow PR.
        """
        # Some model config classes like CLIP define their own `from_pretrained` without the new argument `token` yet.
        if token is None:
            token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

    @classmethod
    def from_pretrained(
        cls: type[SpecificPretrainedConfigType],
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> SpecificPretrainedConfigType:
        r"""
        Instantiate a [`PretrainedConfig`] (or a derived class) from a pretrained model configuration.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a configuration file saved using the
                  [`~PretrainedConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved configuration JSON *file*, e.g., `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        # We can't instantiate directly the base class *PretrainedConfig* so let's show the examples on a
        # derived class: BertConfig
        config = BertConfig.from_pretrained(
            "google-bert/bert-base-uncased"
        )  # Download configuration from huggingface.co and cache.
        config = BertConfig.from_pretrained(
            "./test/saved_model/"
        )  # E.g. config (or model) was saved using *save_pretrained('./test/saved_model/')*
        config = BertConfig.from_pretrained("./test/saved_model/my_configuration.json")
        config = BertConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        assert config.output_attentions == True
        config, unused_kwargs = BertConfig.from_pretrained(
            "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        )
        assert config.output_attentions == True
        assert unused_kwargs == {"foo": False}
        ```"""
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        cls._set_token_in_kwargs(kwargs, token)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if cls.base_config_key and cls.base_config_key in config_dict:
            config_dict = config_dict[cls.base_config_key]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            # sometimes the config has no `base_config_key` if the config is used in several composite models
            # e.g. LlamaConfig. In that case we try to see if there is match in `model_type` before raising a warning
            for v in config_dict.values():
                if isinstance(v, dict) and v.get("model_type") == cls.model_type:
                    config_dict = v

            # raise warning only if we still can't see a match in `model_type`
            if config_dict["model_type"] != cls.model_type:
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        [`PretrainedConfig`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        cls._set_token_in_kwargs(kwargs)

        original_kwargs = copy.deepcopy(kwargs)
        # Get config dict associated with the base config file
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict is None:
            return {}, kwargs
        if "_commit_hash" in config_dict:
            original_kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # That config file may point us toward another config file to use.
        if "configuration_files" in config_dict:
            configuration_file = get_configuration_file(config_dict["configuration_files"])
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )

        return config_dict, kwargs

    @classmethod
    def _get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        gguf_file = kwargs.get("gguf_file")

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            # Special case when pretrained_model_name_or_path is a local file
            resolved_config_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            configuration_file = pretrained_model_name_or_path if gguf_file is None else gguf_file
            resolved_config_file = download_url(pretrained_model_name_or_path)
        else:
            configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME) if gguf_file is None else gguf_file

            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _commit_hash=commit_hash,
                )
                if resolved_config_file is None:
                    return None, kwargs
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory"
                    f" containing a {configuration_file} file"
                )

        try:
            if gguf_file:
                config_dict = load_gguf_checkpoint(resolved_config_file, return_tensors=False)["config"]
            else:
                # Load config dict
                config_dict = cls._dict_from_json_file(resolved_config_file)

            config_dict["_commit_hash"] = commit_hash
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise OSError(f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file.")

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")

        # timm models are not saved with the model_type in the config file
        if "model_type" not in config_dict and is_timm_config_dict(config_dict):
            config_dict["model_type"] = "timm_wrapper"

        return config_dict, kwargs

    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> SpecificPretrainedConfigType:
        """
        Instantiates a [`PretrainedConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the [`~PretrainedConfig.get_config_dict`] method.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # For BC on the old `torch_dtype`
        if (torch_dtype := kwargs.pop("torch_dtype", None)) is not None:
            logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
            # If both are present, use `dtype`
            kwargs["dtype"] = kwargs.get("dtype", torch_dtype)

        # We remove it from kwargs so that it does not appear in `return_unused_kwargs`.
        config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = {int(key): value for key, value in config.pruned_heads.items()}

        # Update config with kwargs if needed
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels}` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                current_attr = getattr(config, key)
                # To authorize passing a custom subconfig as kwarg in models that have nested configs.
                if isinstance(current_attr, PretrainedConfig) and isinstance(value, dict):
                    value = current_attr.__class__(**value)
                setattr(config, key, value)
                if key != "dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Model config {config}")
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(
        cls: type[SpecificPretrainedConfigType], json_file: Union[str, os.PathLike]
    ) -> SpecificPretrainedConfigType:
        """
        Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return isinstance(other, PretrainedConfig) and (self.__dict__ == other.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def __iter__(self):
        yield from self.__dict__

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from the configuration that correspond to the default config attributes for
        better readability, while always retaining the `config` attribute from the class. Serializes to a
        Python dictionary.

        Returns:
            dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        config_dict = self.to_dict()

        # Get the default config dict (from a fresh PreTrainedConfig instance)
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict() if not self.has_no_defaults_at_init else {}

        serializable_config_dict = {}

        # Only serialize values that differ from the default config,
        # except always keep the 'config' attribute.
        for key, value in config_dict.items():
            if (
                isinstance(getattr(self, key, None), PretrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
                or key in self.sub_configs
            ):
                # For nested configs we need to clean the diff recursively
                diff = recursive_diff_dict(value, default_config_dict, config_obj=getattr(self, key, None))
                if "model_type" in value:
                    # Needs to be set even if it's not in the diff
                    diff["model_type"] = value["model_type"]

                serializable_config_dict[key] = diff
            elif (
                key not in default_config_dict
                or key == "transformers_version"
                or key == "vocab_file"
                or value != default_config_dict[key]
                or (key in default_config_dict and value != class_config_dict.get(key, value))
            ):
                serializable_config_dict[key] = value

        self._remove_keys_not_serialized(serializable_config_dict)

        # Key removed only in diff dict
        if "_name_or_path" in serializable_config_dict:
            del serializable_config_dict["_name_or_path"]

        if hasattr(self, "quantization_config"):
            serializable_config_dict["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )
        self.dict_dtype_to_str(serializable_config_dict)

        return serializable_config_dict

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Transformers version when serializing the model
        output["transformers_version"] = __version__

        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, PretrainedConfig):
                value = value.to_dict()
                del value["transformers_version"]

            output[key] = value

        self._remove_keys_not_serialized(output)

        if hasattr(self, "quantization_config"):
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )
        self.dict_dtype_to_str(output)

        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
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
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: dict[str, Any]):
        """
        Updates attributes of this class with attributes from `config_dict`.

        Args:
            config_dict (`dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from `update_str`.

        The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

        The keys to change have to already exist in the config object.

        Args:
            update_str (`str`): String with attributes that should be updated for this class.

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
                raise TypeError(
                    f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
                )

            setattr(self, k, v)

    def dict_dtype_to_str(self, d: dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("dtype") is not None:
            if isinstance(d["dtype"], dict):
                d["dtype"] = {k: str(v).split(".")[-1] for k, v in d["dtype"].items()}
            elif not isinstance(d["dtype"], str):
                d["dtype"] = str(d["dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_dtype_to_str(value)

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        """
        Checks and removes if there are any keys in the dict that should not be serialized when saving the config.
        Runs recursive check on the dict, to remove from all sub configs.
        """
        if hasattr(self, "quantization_config"):
            # Pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            _ = d.pop("_pre_quantization_dtype", None)

        if "_auto_class" in d:
            del d["_auto_class"]
        if "_output_attentions" in d:
            d["output_attentions"] = d.pop("_output_attentions")
        if "_commit_hash" in d:
            del d["_commit_hash"]
        if "_attn_implementation_internal" in d:
            del d["_attn_implementation_internal"]
        # Do not serialize `base_model_tp_plan` for now
        if "base_model_tp_plan" in d:
            del d["base_model_tp_plan"]
        # Do not serialize `base_model_pp_plan` for now
        if "base_model_pp_plan" in d:
            del d["base_model_pp_plan"]
        for value in d.values():
            if isinstance(value, dict):
                self._remove_keys_not_serialized(value)

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoConfig"):
        """
        Register this class with a given auto class. This should only be used for custom configurations as the ones in
        the library are already mapped with `AutoConfig`.



        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoConfig"`):
                The auto class to register this new configuration with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    @staticmethod
    def _get_global_generation_defaults() -> dict[str, Any]:
        return {
            "max_length": 20,
            "min_length": 0,
            "do_sample": False,
            "early_stopping": False,
            "num_beams": 1,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "typical_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "encoder_no_repeat_ngram_size": 0,
            "bad_words_ids": None,
            "num_return_sequences": 1,
            "output_scores": False,
            "return_dict_in_generate": False,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "remove_invalid_values": False,
            "exponential_decay_length_penalty": None,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
        }

    def _get_non_default_generation_parameters(self) -> dict[str, Any]:
        """
        Gets the non-default generation parameters on the PretrainedConfig instance
        """
        non_default_generation_parameters = {}
        decoder_attribute_name = None

        # Composite models don't have a default config, use their decoder config as a fallback for default values
        # If no known pattern is matched, then `default_config = None` -> check against the global generation defaults
        try:
            default_config = self.__class__()
        except ValueError:
            decoder_config = self.get_text_config(decoder=True)
            if decoder_config is not self:
                default_config = decoder_config.__class__()
            else:
                default_config = None

        # If it is a composite model, we want to check the subconfig that will be used for generation
        self_decoder_config = self if decoder_attribute_name is None else getattr(self, decoder_attribute_name)

        for parameter_name, default_global_value in self._get_global_generation_defaults().items():
            if hasattr(self_decoder_config, parameter_name):
                is_default_in_config = is_default_generation_value = None
                parameter_value = getattr(self_decoder_config, parameter_name)
                # Three cases in which is okay for the model config to hold generation config parameters:
                # 1. The parameter is set to `None`, effectively delegating its value to the generation config
                if parameter_value is None:
                    continue
                # 2. If we have a default config, then the instance should hold the same generation defaults
                if default_config is not None:
                    is_default_in_config = parameter_value == getattr(default_config, parameter_name)
                # 3. if we don't have a default config, then the instance should hold the global generation defaults
                else:
                    is_default_generation_value = parameter_value == default_global_value

                is_non_default = (is_default_in_config is False) or (
                    is_default_in_config is None and is_default_generation_value is False
                )
                if is_non_default:
                    non_default_generation_parameters[parameter_name] = getattr(self_decoder_config, parameter_name)

        return non_default_generation_parameters

    def get_text_config(self, decoder=False) -> "PretrainedConfig":
        """
        Returns the config that is meant to be used with text IO. On most models, it is the original config instance
        itself. On specific composite models, it is under a set of valid names.

        Args:
            decoder (`Optional[bool]`, *optional*, defaults to `False`):
                If set to `True`, then only search for decoder config names.
        """
        decoder_possible_text_config_names = ("decoder", "generator", "text_config")
        encoder_possible_text_config_names = ("text_encoder",)
        if decoder:
            possible_text_config_names = decoder_possible_text_config_names
        else:
            possible_text_config_names = encoder_possible_text_config_names + decoder_possible_text_config_names

        valid_text_config_names = []
        for text_config_name in possible_text_config_names:
            if hasattr(self, text_config_name):
                text_config = getattr(self, text_config_name, None)
                if text_config is not None:
                    valid_text_config_names += [text_config_name]

        if len(valid_text_config_names) > 1:
            raise ValueError(
                f"Multiple valid text configs were found in the model config: {valid_text_config_names}. In this "
                "case, using `get_text_config()` would be ambiguous. Please specify the desied text config directly."
            )
        elif len(valid_text_config_names) == 1:
            config_to_return = getattr(self, valid_text_config_names[0])
        else:
            config_to_return = self
        return config_to_return

    @classmethod
    def from_text_vision_configs(cls, text_config, vision_config, **kwargs):
        r"""
        Instantiate a model config (or a derived class) from text model configuration and vision model
        configuration.

        Returns:
            [`PreTrainedConfig`]: An instance of a configuration object
        """

        warnings.warn(
            "The `from_text_vision_configs` method is deprecated and will be removed in v4.60 of Transformers. Please instantiate "
            "the config class directly with `MyConfig(text_config=text_config, vision_config=vision_config, **kwargs)` instead.",
            FutureWarning,
        )

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

    @classmethod
    def from_text_audio_configs(cls, text_config, audio_config, **kwargs):
        r"""
        Instantiate a model config (or a derived class) from text model configuration and audio model
        configuration.

        Returns:
            [`PreTrainedConfig`]: An instance of a configuration object
        """

        warnings.warn(
            "The `from_text_audio_configs` method is deprecated and will be removed in v4.60 of Transformers. Please instantiate "
            "the config class directly with `MyConfig(text_config=text_config, audio_config=audio_config, **kwargs)` instead.",
            FutureWarning,
        )

        return cls(text_config=text_config.to_dict(), audio_config=audio_config.to_dict(), **kwargs)


def get_configuration_file(configuration_files: list[str]) -> str:
    """
    Get the configuration file to use for this version of transformers.

    Args:
        configuration_files (`list[str]`): The list of available configuration files.

    Returns:
        `str`: The configuration file to use.
    """
    configuration_files_map = {}
    for file_name in configuration_files:
        if file_name.startswith("config.") and file_name.endswith(".json") and file_name != "config.json":
            v = file_name.removeprefix("config.").removesuffix(".json")
            configuration_files_map[v] = file_name
    available_versions = sorted(configuration_files_map.keys())

    # Defaults to FULL_CONFIGURATION_FILE and then try to look at some newer versions.
    configuration_file = CONFIG_NAME
    transformers_version = version.parse(__version__)
    for v in available_versions:
        if version.parse(v) <= transformers_version:
            configuration_file = configuration_files_map[v]
        else:
            # No point going further since the versions are sorted.
            break

    return configuration_file


def recursive_diff_dict(dict_a, dict_b, config_obj=None):
    """
    Helper function to recursively take the diff between two nested dictionaries. The resulting diff only contains the
    values from `dict_a` that are different from values in `dict_b`.

    dict_b : the default config dictionary. We want to remove values that are in this one
    """
    diff = {}
    default = config_obj.__class__().to_dict() if config_obj is not None else {}
    for key, value in dict_a.items():
        obj_value = getattr(config_obj, str(key), None)
        if isinstance(obj_value, PretrainedConfig) and key in dict_b and isinstance(dict_b[key], dict):
            diff_value = recursive_diff_dict(value, dict_b[key], config_obj=obj_value)
            diff[key] = diff_value
        elif key not in dict_b or (value != default[key]):
            diff[key] = value
    return diff


PretrainedConfig.push_to_hub = copy_func(PretrainedConfig.push_to_hub)
if PretrainedConfig.push_to_hub.__doc__ is not None:
    PretrainedConfig.push_to_hub.__doc__ = PretrainedConfig.push_to_hub.__doc__.format(
        object="config", object_class="AutoConfig", object_files="configuration file"
    )


ALLOWED_LAYER_TYPES = (
    "full_attention",
    "sliding_attention",
    "chunked_attention",
    "linear_attention",  # used in minimax
)


def layer_type_validation(layer_types: list[str]):
    """Check that each entry in `layer_types` are allowed."""
    if not all(layer_type in ALLOWED_LAYER_TYPES for layer_type in layer_types):
        raise ValueError(f"The `layer_types` entries must be in {ALLOWED_LAYER_TYPES}")
