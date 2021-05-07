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
""" AutoFeatureExtractor class. """

import os
from collections import OrderedDict

from transformers import DeiTFeatureExtractor, Speech2TextFeatureExtractor, ViTFeatureExtractor

from ... import DeiTConfig, PretrainedConfig, Speech2TextConfig, ViTConfig, Wav2Vec2Config
from ...feature_extraction_utils import FeatureExtractionMixin

# Build the list of all feature extractors
from ...file_utils import FEATURE_EXTRACTOR_NAME
from ..wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from .configuration_auto import AutoConfig, replace_list_option_in_docstrings


FEATURE_EXTRACTOR_MAPPING = OrderedDict(
    [
        (DeiTConfig, DeiTFeatureExtractor),
        (Speech2TextConfig, Speech2TextFeatureExtractor),
        (ViTConfig, ViTFeatureExtractor),
        (Wav2Vec2Config, Wav2Vec2FeatureExtractor),
    ]
)


def feature_extractor_class_from_name(class_name: str):
    for c in FEATURE_EXTRACTOR_MAPPING.values():
        if c is not None and c.__name__ == class_name:
            return c


class AutoFeatureExtractor:
    r"""
    This is a generic feature extractor class that will be instantiated as one of the feature extractor classes of the
    library when created with the :meth:`AutoFeatureExtractor.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoFeatureExtractor is designed to be instantiated "
            "using the `AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(FEATURE_EXTRACTOR_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the feature extractor classes of the library from a pretrained model vocabulary.

        The tokenizer class to instantiate is selected based on the :obj:`model_type` property of the config object
        (either passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a feature extractor file saved using the
                  :func:`~transformers.feature_extraction_utils.FeatureExtractionMixin.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - a path or url to a saved feature extractor JSON `file`, e.g.,
                  ``./my_model_directory/preprocessor_config.json``.
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
                controlled by the ``return_unused_kwargs`` keyword parameter.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.

        Examples::

            >>> from transformers import AutoFeatureExtractor

            >>> # Download vocabulary from huggingface.co and cache.
            >>> feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')

            >>> # If vocabulary files are in a directory (e.g. feature extractor was saved using `save_pretrained('./test/saved_model/')`)
            >>> feature_extractor = AutoFeatureExtractor.from_pretrained('./test/saved_model/')

        """
        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True

        is_feature_extraction_file = os.path.isfile(pretrained_model_name_or_path)
        is_directory = os.path.isdir(pretrained_model_name_or_path) and os.path.exists(
            os.path.join(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME)
        )

        if not is_feature_extraction_file and not is_directory:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        kwargs["_from_auto"] = True
        config_dict, _ = FeatureExtractionMixin.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)

        if type(config) in FEATURE_EXTRACTOR_MAPPING.keys():
            return FEATURE_EXTRACTOR_MAPPING[type(config)].from_dict(config_dict, **kwargs)
        elif "feature_extractor_type" in config_dict:
            feature_extractor_class = feature_extractor_class_from_name(config_dict["feature_extractor_type"])
            return feature_extractor_class.from_dict(config_dict, **kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. Should have a `feature_extractor_type` key in "
            f"its {FEATURE_EXTRACTOR_NAME}, or contain one of the following strings "
            f"in its name: {', '.join(FEATURE_EXTRACTOR_MAPPING.keys())}"
        )
