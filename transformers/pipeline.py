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
    def __init__(self, tokenizer, model, is_compiled=False, is_trained=False):
        self.tokenizer = tokenizer
        self.model = model
        if is_tf_available():
            self.framework = 'tf'
        elif is_torch_available():
            self.framework = 'pt'
        else:
            raise ImportError("At least one of PyTorch or TensorFlow 2.0+ should be installed to use CLI training")
        self.is_compiled = is_compiled
        self.is_trained = is_trained


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiates a pipeline from a pre-trained tokenizer and model.
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

        model_kwargs['output_loading_info'] = True
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, **model_kwargs)

        return cls(tokenizer, model, is_trained=bool(not loading_info['missing_keys']))


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
        self.is_trained = True


    def __call__(self, text):
        if not self.is_trained:
            logger.error("Some weights of the model are not trained. Please fine-tune the model on a classification task before using it.")

        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors=self.framework)
        if self.framework == 'tf':
            # TODO trace model
            predictions = self.model(**inputs)[0]
        else:
            with torch.no_grad():
                predictions = self.model(**inputs)[0]

        return predictions.numpy().tolist()
