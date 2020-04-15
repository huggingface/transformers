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
""" Configuration base class and utilities."""


import copy
import json
import logging
import os

from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    TF2_WEIGHTS_NAME,
    WEIGHTS_NAME,
    cached_path,
    hf_bucket_url,
    is_remote_url,
)


logger = logging.getLogger(__name__)


class ModelCard:
    r""" Structured Model Card class.
        Store model card as well as methods for loading/downloading/saving model cards.

        Please read the following paper for details and explanation on the sections:
            "Model Cards for Model Reporting"
                by Margaret Mitchell, Simone Wu,
                Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer,
                Inioluwa Deborah Raji and Timnit Gebru for the proposal behind model cards.
            Link: https://arxiv.org/abs/1810.03993

        Note:
            A model card can be loaded and saved to disk.

        Parameters:
    """

    def __init__(self, **kwargs):
        # Recomended attributes from https://arxiv.org/abs/1810.03993 (see papers)
        self.model_details = kwargs.pop("model_details", {})
        self.intended_use = kwargs.pop("intended_use", {})
        self.factors = kwargs.pop("factors", {})
        self.metrics = kwargs.pop("metrics", {})
        self.evaluation_data = kwargs.pop("evaluation_data", {})
        self.training_data = kwargs.pop("training_data", {})
        self.quantitative_analyses = kwargs.pop("quantitative_analyses", {})
        self.ethical_considerations = kwargs.pop("ethical_considerations", {})
        self.caveats_and_recommendations = kwargs.pop("caveats_and_recommendations", {})

        # Open additional attributes
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    def save_pretrained(self, save_directory_or_file):
        """ Save a model card object to the directory or file `save_directory_or_file`.
        """
        if os.path.isdir(save_directory_or_file):
            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_card_file = os.path.join(save_directory_or_file, MODEL_CARD_NAME)
        else:
            output_model_card_file = save_directory_or_file

        self.to_json_file(output_model_card_file)
        logger.info("Model card saved in {}".format(output_model_card_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiate a :class:`~transformers.ModelCard` from a pre-trained model model card.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model card to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model card that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a model card file saved using the :func:`~transformers.ModelCard.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - a path or url to a saved model card JSON `file`, e.g.: ``./my_model_directory/modelcard.json``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                card should be cached if the standard cache should not be used.

            kwargs: (`optional`) dict: key/value pairs with which to update the ModelCard object after loading.

                - The values in kwargs of any keys which are model card attributes will be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* model card attributes is controlled by the `return_unused_kwargs` keyword parameter.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            find_from_standard_name: (`optional`) boolean, default True:
                If the pretrained_model_name_or_path ends with our standard model or config filenames, replace them with our standard modelcard filename.
                Can be used to directly feed a model/config url and access the colocated modelcard.

            return_unused_kwargs: (`optional`) bool:

                - If False, then this function returns just the final model card object.
                - If True, then this functions returns a tuple `(model card, unused_kwargs)` where `unused_kwargs` is a dictionary consisting of the key/value pairs whose keys are not model card attributes: ie the part of kwargs which has not been used to update `ModelCard` and is otherwise ignored.

        Examples::

            modelcard = ModelCard.from_pretrained('bert-base-uncased')    # Download model card from S3 and cache.
            modelcard = ModelCard.from_pretrained('./test/saved_model/')  # E.g. model card was saved using `save_pretrained('./test/saved_model/')`
            modelcard = ModelCard.from_pretrained('./test/saved_model/modelcard.json')
            modelcard = ModelCard.from_pretrained('bert-base-uncased', output_attention=True, foo=False)

        """
        cache_dir = kwargs.pop("cache_dir", None)
        proxies = kwargs.pop("proxies", None)
        find_from_standard_name = kwargs.pop("find_from_standard_name", True)
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        if pretrained_model_name_or_path in ALL_PRETRAINED_CONFIG_ARCHIVE_MAP:
            # For simplicity we use the same pretrained url than the configuration files
            # but with a different suffix (modelcard.json). This suffix is replaced below.
            model_card_file = ALL_PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            model_card_file = os.path.join(pretrained_model_name_or_path, MODEL_CARD_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            model_card_file = pretrained_model_name_or_path
        else:
            model_card_file = hf_bucket_url(pretrained_model_name_or_path, postfix=MODEL_CARD_NAME)

        if find_from_standard_name or pretrained_model_name_or_path in ALL_PRETRAINED_CONFIG_ARCHIVE_MAP:
            model_card_file = model_card_file.replace(CONFIG_NAME, MODEL_CARD_NAME)
            model_card_file = model_card_file.replace(WEIGHTS_NAME, MODEL_CARD_NAME)
            model_card_file = model_card_file.replace(TF2_WEIGHTS_NAME, MODEL_CARD_NAME)

        try:
            # Load from URL or cache if already cached
            resolved_model_card_file = cached_path(
                model_card_file, cache_dir=cache_dir, force_download=True, proxies=proxies, resume_download=False
            )
            if resolved_model_card_file is None:
                raise EnvironmentError
            if resolved_model_card_file == model_card_file:
                logger.info("loading model card file {}".format(model_card_file))
            else:
                logger.info(
                    "loading model card file {} from cache at {}".format(model_card_file, resolved_model_card_file)
                )
            # Load model card
            modelcard = cls.from_json_file(resolved_model_card_file)

        except (EnvironmentError, json.JSONDecodeError):
            # We fall back on creating an empty model card
            modelcard = cls()

        # Update model card with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(modelcard, key):
                setattr(modelcard, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model card: %s", str(modelcard))
        if return_unused_kwargs:
            return modelcard, kwargs
        else:
            return modelcard

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `ModelCard` from a Python dictionary of parameters."""
        return cls(**json_object)

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `ModelCard` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        dict_obj = json.loads(text)
        return cls(**dict_obj)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
