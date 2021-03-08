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
"""
 Feature extraction saving/loading class for common feature extractors.
"""

import copy
import json
import os
from typing import Any, Dict, Tuple, Union

from .file_utils import FEATURE_EXTRACTOR_NAME, cached_path, hf_bucket_url, is_remote_url
from .utils import logging


logger = logging.get_logger(__name__)

PreTrainedFeatureExtractor = Union["PreTrainedSequenceFeatureExtractor"]  # noqa: F821


class FeatureExtractionSavingUtilsMixin:
    """
    This is a feature extraction mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    """

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PreTrainedFeatureExtractor":
        r"""
        Instantiate a :class:`~transformers.PreTrainedSequenceFeatureExtractor` (or a derived class) from a pretrained
        feature extractor.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a feature extractor file saved using the
                  :func:`~transformers.PreTrainedSequenceFeatureExtractor.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - a path or url to a saved feature extractor JSON `file`, e.g.,
                  ``./my_model_directory/feature_extraction_config.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the feature extractor files and override the cached versions
                if they exist.
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
                If :obj:`False`, then this function returns just the final feature extractor object. If :obj:`True`,
                then this functions returns a :obj:`Tuple(feature_extractor, unused_kwargs)` where `unused_kwargs` is a
                dictionary consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the
                part of ``kwargs`` which has not been used to update ``feature_extractor`` and is otherwise ignored.
            kwargs (:obj:`Dict[str, Any]`, `optional`):
                The values in kwargs of any keys which are feature extractor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
                controlled by the ``return_unused_kwargs`` keyword parameter. .. note::
            Passing :obj:`use_auth_token=True` is required when you want to use a private model.

        Returns:
            :class:`~transformers.PreTrainedSequenceFeatureExtractor`: The feature extractor object instantiated from
            this pretrained model.

        Examples::
            # We can't instantiate directly the base class `PreTrainedSequenceFeatureExtractor` so let's show the examples on a
            # derived class: Wav2Vec2FeatureExtractor
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')    # Download feature_extraction_config from huggingface.co and cache.
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('./test/saved_model/')  # E.g. feature_extractor (or model) was saved using `save_pretrained('./test/saved_model/')`
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('./test/saved_model/preprocessor_config.json')
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h', return_attention_mask=False, foo=False)
            assert feature_extractor.return_attention_mask is False
            feature_extractor, unused_kwargs = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h', return_attention_mask=False,
                                                               foo=False, return_unused_kwargs=True)
            assert feature_extractor.return_attention_mask is False
            assert unused_kwargs == {'foo': False}
        """
        feature_extractor_dict, kwargs = cls.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(feature_extractor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save a feature_extractor object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.PreTrainedSequenceFeatureExtractor.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature extractor JSON file will be saved (will be created if it does not exist).
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_feature_extractor_file = os.path.join(save_directory, FEATURE_EXTRACTOR_NAME)

        self.to_json_file(output_feature_extractor_file)
        logger.info(f"Configuration saved in {output_feature_extractor_file}")

    @classmethod
    def get_feature_extractor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        :class:`~transformers.PreTrainedSequenceFeatureExtractor` using ``from_dict``.

        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor
            object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            feature_extractor_file = os.path.join(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            feature_extractor_file = pretrained_model_name_or_path
        else:
            feature_extractor_file = hf_bucket_url(
                pretrained_model_name_or_path, filename=FEATURE_EXTRACTOR_NAME, revision=revision, mirror=None
            )

        try:
            # Load from URL or cache if already cached
            resolved_feature_extractor_file = cached_path(
                feature_extractor_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
            )
            # Load feature_extractor dict
            with open(resolved_feature_extractor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            feature_extractor_dict = json.loads(text)

        except EnvironmentError as err:
            logger.error(err)
            msg = (
                f"Can't load feature extractor for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {FEATURE_EXTRACTOR_NAME} file\n\n"
            )
            raise EnvironmentError(msg)

        except json.JSONDecodeError:
            msg = (
                f"Couldn't reach server at '{feature_extractor_file}' to download feature extractor configuration file or "
                "feature extractor configuration file is not a valid JSON file. "
                f"Please check network or file content here: {resolved_feature_extractor_file}."
            )
            raise EnvironmentError(msg)

        if resolved_feature_extractor_file == feature_extractor_file:
            logger.info(f"loading feature extractor configuration file {feature_extractor_file}")
        else:
            logger.info(
                f"loading feature extractor configuration file {feature_extractor_file} from cache at {resolved_feature_extractor_file}"
            )

        return feature_extractor_dict, kwargs

    @classmethod
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> PreTrainedFeatureExtractor:
        """
        Instantiates a :class:`~transformers.PreTrainedSequenceFeatureExtractor` from a Python dictionary of
        parameters.

        Args:
            feature_extractor_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.PreTrainedSequenceFeatureExtractor.to_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            :class:`~transformers.PreTrainedSequenceFeatureExtractor`: The feature extractor object instantiated from
            those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        feature_extractor = cls(**feature_extractor_dict)

        # Update feature_extractor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(feature_extractor, key):
                setattr(feature_extractor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Feature extractor {feature_extractor}")
        if return_unused_kwargs:
            return feature_extractor, kwargs
        else:
            return feature_extractor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this feature extractor instance.
        """
        output = copy.deepcopy(self.__dict__)

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> PreTrainedFeatureExtractor:
        """
        Instantiates a :class:`~transformers.PreTrainedSequenceFeatureExtractor` from the path to a JSON file of
        parameters.

        Args:
            json_file (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`~transformers.PreTrainedSequenceFeatureExtractor`: The feature_extractor object instantiated from
            that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        feature_extractor_dict = json.loads(text)
        return cls(**feature_extractor_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this feature_extractor instance in JSON
            format.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this feature_extractor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"
