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
""" Pipeline class: Tokenizer + Model. """

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import logging

from .modeling_auto import (AutoModel, AutoModelForQuestionAnswering,
                            AutoModelForSequenceClassification,
                            AutoModelWithLMHead)
from .tokenization_auto import AutoTokenizer
from .file_utils import add_start_docstrings, is_tf_available, is_torch_available
from .data.processors import SingleSentenceClassificationProcessor

if is_tf_available():
    import tensorflow as tf
if is_torch_available():
    import torch

logger = logging.getLogger(__name__)

# TF training parameters
USE_XLA = False
USE_AMP = False

class TextClassificationPipeline(object):
    r"""
        :class:`~transformers.TextClassificationPipeline` is a class encapsulating a pretrained model and
        its tokenizer and will be instantiated as one of the base model classes of the library
        when created with the `Pipeline.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertModel (DistilBERT model)
            - contains `roberta`: RobertaModel (RoBERTa model)
            - contains `bert`: BertModel (Bert model)
            - contains `openai-gpt`: OpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: GPT2Model (OpenAI GPT-2 model)
            - contains `ctrl`: CTRLModel (Salesforce CTRL  model)
            - contains `transfo-xl`: TransfoXLModel (Transformer-XL model)
            - contains `xlnet`: XLNetModel (XLNet model)
            - contains `xlm`: XLMModel (XLM model)
    """
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        if is_tf_available():
            self.framework = 'tf'
        elif is_torch_available():
            self.framework = 'pt'
        else:
            raise ImportError("At least one of PyTorch or TensorFlow 2.0+ should be installed to use CLI training")
        self.is_compiled = False


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiates one of the base model classes of the library
        from a pre-trained model configuration.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertModel (DistilBERT model)
            - contains `roberta`: RobertaModel (RoBERTa model)
            - contains `bert`: BertModel (Bert model)
            - contains `openai-gpt`: OpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: GPT2Model (OpenAI GPT-2 model)
            - contains `ctrl`: CTRLModel (Salesforce CTRL  model)
            - contains `transfo-xl`: TransfoXLModel (Transformer-XL model)
            - contains `xlnet`: XLNetModel (XLNet model)
            - contains `xlm`: XLMModel (XLM model)

            The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
            To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

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

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModel.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModel.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        # Extract tokenizer and model arguments
        tokenizer_kwargs = {}
        for key in kwargs:
            if key.startswith('tokenizer_'):
                # Specific to the tokenizer
                tokenizer_kwargs[key.replace('tokenizer_', '')] = kwargs.pop(key)
            elif not key.startswith('model_'):
                # used for both the tokenizer and the model
                tokenizer_kwargs[key] = kwargs[key]

        model_kwargs = {}
        for key in kwargs:
            if key.startswith('model_'):
                # Specific to the model
                model_kwargs[key.replace('model_', '')] = kwargs.pop(key)
            elif not key.startswith('tokenizer_'):
                # used for both the tokenizer and the model
                model_kwargs[key] = kwargs[key]

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
        return cls(tokenizer, model)


    def save_pretrained(self, save_directory):
        if not os.path.isdir(save_directory):
            logger.error("Saving directory ({}) should be a directory".format(save_directory))
            return
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)


    def compile(self, learning_rate=3e-5, epsilon=1e-8):
        if self.framework == 'tf':
            logger.info('Preparing model')
            # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
            if USE_AMP:
                # loss scaling is currently required when using mixed precision
                opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
            self.model.compile(optimizer=opt, loss=loss, metrics=[metric])
        else:
            raise NotImplementedError
        self.is_compiled = True


    def prepare_data(self, train_samples_text, train_samples_labels,
                     valid_samples_text=None, valid_samples_labels=None,
                     validation_split=0.1):
        dataset = SingleSentenceClassificationProcessor.create_from_examples(train_samples_text,
                                                                             train_samples_labels)
        num_data_samples = len(dataset)
        if valid_samples_text is not None and valid_samples_labels is not None:
            valid_dataset = SingleSentenceClassificationProcessor.create_from_examples(valid_samples_text,
                                                                                       valid_samples_labels)
            num_valid_samples = len(valid_dataset)
            train_dataset = dataset
            num_train_samples = num_data_samples
        else:
            assert 0.0 < validation_split < 1.0, "validation_split should be between 0.0 and 1.0"
            num_valid_samples = int(num_data_samples * validation_split)
            num_train_samples = num_data_samples - num_valid_samples
            train_dataset = dataset[num_train_samples]
            valid_dataset = dataset[num_valid_samples]

        logger.info('Tokenizing and processing dataset')
        train_dataset = train_dataset.get_features(self.tokenizer, return_tensors=self.framework)
        valid_dataset = valid_dataset.get_features(self.tokenizer, return_tensors=self.framework)
        return train_dataset, valid_dataset, num_train_samples, num_valid_samples


    def fit(self, train_samples_text, train_samples_labels,
            valid_samples_text=None, valid_samples_labels=None,
            train_batch_size=None, valid_batch_size=None,
            validation_split=0.1,
            **kwargs):

        if not self.is_compiled:
            self.compile()

        datasets = self.prepare_data(train_samples_text, train_samples_labels,
                                     valid_samples_text, valid_samples_labels,
                                     validation_split)
        train_dataset, valid_dataset, num_train_samples, num_valid_samples = datasets

        train_steps = num_train_samples//train_batch_size
        valid_steps = num_valid_samples//valid_batch_size

        if self.framework == 'tf':
            # Prepare dataset as a tf.train_data.Dataset instance
            train_dataset = train_dataset.shuffle(128).batch(train_batch_size).repeat(-1)
            valid_dataset = valid_dataset.batch(valid_batch_size)

            logger.info('Training TF 2.0 model')
            history = self.model.fit(train_dataset, epochs=2, steps_per_epoch=train_steps,
                                     validation_data=valid_dataset, validation_steps=valid_steps, **kwargs)
        else:
            raise NotImplementedError


    def __call__(self, text):
        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors=self.framework)
        if self.framework == 'tf':
            # TODO trace model
            predictions = self.model(**inputs)[0]
        else:
            with torch.no_grad():
                predictions = self.model(**inputs)[0]

        return predictions.numpy().tolist()
