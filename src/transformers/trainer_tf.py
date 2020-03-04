# coding=utf-8
""" Trainer class."""

import os
import logging
from collections import defaultdict, OrderedDict
import math
import itertools

import numpy as np

from sklearn.metrics import classification_report

import tensorflow as tf
from .optimization_tf import *
from .losses_tf import *
from .configuration_auto import AutoConfig
from .tokenization_auto import AutoTokenizer
from .data.processors.glue import *
from .data.processors.squad import *
from .data.processors.xnli import *
from .data.processors.utils import *
from .modeling_tf_albert import *
from .modeling_tf_bert import *
from .modeling_tf_camembert import *
from .modeling_tf_ctrl import *
from .modeling_tf_distilbert import *
from .modeling_tf_gpt2 import *
from .modeling_tf_openai import *
from .modeling_tf_roberta import *
from .modeling_tf_t5 import *
from .modeling_tf_transfo_xl import *
from .modeling_tf_xlm import *
from .modeling_tf_xlm_roberta import *
from .modeling_tf_xlnet import *


logger = logging.getLogger(__name__)


class TFTrainer(object):
    def __init__(self, **kwargs):
        self.pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        self.optimizer_name = kwargs.pop("optimizer_name", None)
        self.warmup_steps = kwargs.pop("warmup_steps", None)
        self.decay_steps = kwargs.pop("decay_steps", None)
        self.learning_rate = kwargs.pop("learning_rate", None)
        self.epsilon = kwargs.pop("epsilon", None)
        self.loss_name = kwargs.pop("loss_name", None)
        self.batch_size = kwargs.pop("batch_size", None)
        self.eval_batch_size = kwargs.pop("eval_batch_size", None)
        self.distributed = kwargs.pop("distributed", None)
        self.epochs = kwargs.pop("epochs", None)
        self.data_processor_name = kwargs.pop("data_processor_name", None)
        self.task = kwargs.pop("task", None)
        self.architecture = kwargs.pop("architecture", None)
        self.max_len = kwargs.pop("max_len", None)
        self.language = kwargs.pop("language", None)
        self.train_language = kwargs.pop("train_language", None)
        self.doc_stride = kwargs.pop("doc_stride", None)
        self.max_query_len = kwargs.pop("max_query_len", None)
        self.max_grad_norm = kwargs.pop("max_grad_norm", None)
        self.labels = kwargs.pop("labels", None)
        self.metric_name = kwargs.pop("metric_name", None)
        self.datasets = {}

        assert len(kwargs) == 0, "unrecognized params passed: %s" % ",".join(kwargs.keys())
    
    def setup_training(self, checkpoint_path, log_path, cache_dir="cache", data_dir=None, training_data=None, validation_data=None, test_data=None):
        """
        Setup the different steps to train a model:
          - check if all the data are given
          - create the proper strategy
          - create the features
          - prepare the model settings
        
        Args:
          checkpoint_path: the directory path where the model checkpoints will be saved.
          log_path: the directory path where the Tensorboard logs will be saved.
          cache_dir (optional): the directory path where the pretrained model and data will be cached.
            "./cache" folder by default.
          data_dir (optional): the directoty path where the data are. This parameter becomes mandatory if
            the parameters training_data and validation_data are not given.
          training_data (optional): the training data, either as a list of example or as a tf.data.Dataset.
            This parameter becomes mandatory if the parameter data_dir is not given.
          validation_data (optional): the validation data, either as a list of example or as a tf.data.Dataset.
            This parameter becomes mandatory if the parameter data_dir is not given.
          test_data (optional): the test data, either as a list of example or as a tf.data.Dataset.
            This parameter will take the validation data as value if nothing is given.
        """

        if data_dir is None and (training_data is None or validation_data is None):
            raise ValueError(
              "If the data_dir parameter is None, the parameters training_data and validation_data should be given"
            )
        elif data_dir is not None and (training_data is not None or validation_data is not None):
            raise ValueError(
              "If the data_dir parameter is not None, the parameters training_data and validation_data should be None"
            )
        elif data_dir is None and (not isinstance(training_data, tf.data.Dataset) or not isinstance(validation_data, tf.data.Dataset) or (test_data is not None and not isinstance(test_data, tf.data.Dataset))):
            raise ValueError(
              "The parameters training_data, validation_data and test_data must be in tf.data.Dataset format"
            )
        
        self.datasets["train"] = training_data
        self.datasets["validation"] = validation_data
        self.datasets["test"] = test_data

        if self.distributed:
            self.strategy = tf.distribute.MirroredStrategy()
            self.batch_size *= self.strategy.num_replicas_in_sync
            self.eval_batch_size *= self.strategy.num_replicas_in_sync
        else:
            if len(tf.config.list_physical_devices('GPU')) >= 1:
                self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            else:
                self.strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path, cache_dir=cache_dir)
        
        if self.data_processor_name == "glue":
            self.processor = glue_processors[self.task]()
        elif self.data_processor_name == "squadv1":
            self.processor = SquadV1Processor()
        elif self.data_processor_name == "squadv2":
            self.processor = SquadV2Processor()
        elif self.data_processor_name == "xnli":
            self.processor = xnli_processors[self.task](language=self.language, train_language=self.train_language)
        else:
            raise ValueError(
                "Allowed data processors are glue, squadv1, squadv2 or xnli. Given: {}".format(self.data_processor_name)
            )

        self._preprocessing_data(tokenizer, data_dir, cache_dir)
        
        try:
            if self.labels is None:
                self.labels = self.processor.get_labels()
                
            self.label2id = {label: i for i, label in enumerate(self.labels)}
            self.id2label = {i: label for i, label in enumerate(self.labels)}
            config = AutoConfig.from_pretrained(self.pretrained_model_name_or_path, num_labels=len(self.labels), id2label=self.id2label, label2id=self.label2id, cache_dir=cache_dir)
        except NotImplementedError:
            self.labels = []
            self.label2id = {}
            self.id2label = {}
            config = AutoConfig.from_pretrained(self.pretrained_model_name_or_path, cache_dir=cache_dir)

        with self.strategy.scope():
            self.model = globals()[self.architecture].from_pretrained(self.pretrained_model_name_or_path, config=config, cache_dir=cache_dir)
            self._create_optimizer()
            self._get_loss_and_metric()
            self._create_checkpoint_manager(checkpoint_path)
            self._create_summary_writer(log_path)
    
    def get_labels(self):
        """
        Returns the list of labels associated to the trained model.
        """
        return self.labels
    
    def _get_loss_and_metric(self):
        """
        Use the loss corresponding to the loss_name field.
        """
        if self.loss_name == "mean_squared_error":
            self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        elif self.loss_name == "sparse_categorical_crossentropy":
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
        elif self.loss_name == "squad_loss":
            self.loss = SquadLoss()
        else:
            raise ValueError(
                "Allowed losses are mean_squared_error or sparse_categorical_crossentropy. Given: {}".format(self.loss_name)
            )
        
        if self.metric_name == "sparse_categorical_accuracy":
            self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        elif self.data_processor_name.startswith("squad"):
            pass
        else:
            raise ValueError(
                "Allowed metrics are sparse_categorical_accuracy. Given: {}".format(self.metric_name)
            )
        
        self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    
    def _create_summary_writer(self, log_path):
        """
        Create a summary writer to be able to read the logs in Tensorboard.
        """
        self.log_path = log_path
        self.train_writer = tf.summary.create_file_writer(log_path + "/train")
        self.test_writer = tf.summary.create_file_writer(log_path + "/test")
    
    def _create_features(self, tokenizer, data_dir):
        """
        Create the features for the training and validation data.

        Args:
          tokenizer: the tokenizer used for encoding the textual data into features.
          data_dir: the directory path where the data are.
          cache_dir: the directory path where the cached data are.
        """
        if self.data_processor_name == "glue":
            if self.datasets["train"] is None:
                train_examples = self.processor.get_train_examples(data_dir)
                self.datasets["train"] = glue_convert_examples_to_features(train_examples, tokenizer, self.max_len, self.task, return_dataset="tf")
            else:
                self.datasets["train"] = glue_convert_examples_to_features(self.datasets["train"], tokenizer, self.max_len, self.task, return_dataset="tf")
            
            if self.datasets["validation"] is None:
                validation_examples = self.processor.get_dev_examples(data_dir)
                self.datasets["validation"] = glue_convert_examples_to_features(validation_examples, tokenizer, self.max_len, self.task, return_dataset="tf")
            else:
                self.datasets["validation"] = glue_convert_examples_to_features(self.datasets["validation"], tokenizer, self.max_len, self.task, return_dataset="tf")

            if self.datasets["test"] is None:
                self.datasets["test"] = self.datasets["validation"]
            else:
                self.datasets["test"] = glue_convert_examples_to_features(self.datasets["test"], tokenizer, self.max_len, self.task, return_dataset="tf")
        elif self.data_processor_name.startswith("squad"):
            if self.datasets["train"] is None:
                train_examples = self.processor.get_train_examples(data_dir)
                self.datasets["train"] = squad_convert_examples_to_features(train_examples, tokenizer, self.max_len, self.doc_stride, self.max_query_len, True, return_dataset="tf")
            else:
                train_examples = self.processor.get_examples_from_dataset(self.datasets["train"])
                self.datasets["train"] = squad_convert_examples_to_features(train_examples, tokenizer, self.max_len, self.doc_stride, self.max_query_len, True, return_dataset="tf")
                
            if self.datasets["validation"] is None:
                validation_examples = self.processor.get_dev_examples(data_dir)
                self.datasets["validation"] = squad_convert_examples_to_features(validation_examples, tokenizer, self.max_len, self.doc_stride, self.max_query_len, False, return_dataset="tf")
            else:
                validation_examples = self.processor.get_examples_from_dataset(self.datasets["validation"], evaluate=True)
                self.datasets["validation"] = squad_convert_examples_to_features(validation_examples, tokenizer, self.max_len, self.doc_stride, self.max_query_len, False, return_dataset="tf")
                
            if self.datasets["test"] is None:
                self.datasets["test"] = self.datasets["validation"]
            else:
                test_examples = self.processor.get_examples_from_dataset(self.datasets["test"])
                self.datasets["test"] = squad_convert_examples_to_features(test_examples, tokenizer, self.max_len, self.doc_stride, self.max_query_len, False, return_dataset="tf")
        elif self.data_processor_name == "xnli":
            if self.datasets["train"] is None:
                train_examples = self.processor.get_train_examples(data_dir)
                self.datasets["train"] = glue_convert_examples_to_features(train_examples, tokenizer, max_length=self.max_len, label_list=self.processor.get_labels(), output_mode=xnli_output_modes[self.task], pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], return_dataset="tf")
            else:
                self.datasets["train"] = glue_convert_examples_to_features(self.datasets["train"], tokenizer, max_length=self.max_len, label_list=self.processor.get_labels(), output_mode=xnli_output_modes[self.task], pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], return_dataset="tf")
            
            if self.datasets["validation"] is None:
                validation_examples = self.processor.get_test_examples(data_dir)
                self.datasets["validation"] = glue_convert_examples_to_features(validation_examples, tokenizer, max_length=self.max_len, label_list=self.processor.get_labels(), output_mode=xnli_output_modes[self.task], pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], return_dataset="tf")
            else:
                self.datasets["validation"] = glue_convert_examples_to_features(self.datasets["validation"], tokenizer, max_length=self.max_len, label_list=self.processor.get_labels(), output_mode=xnli_output_modes[self.task], pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], return_dataset="tf")
            
            if self.datasets["test"] is None:
                self.datasets["test"] = self.datasets["validation"]
            else:
                self.datasets["test"] = glue_convert_examples_to_features(self.datasets["test"], tokenizer, max_length=self.max_len, label_list=self.processor.get_labels(), output_mode=xnli_output_modes[self.task], pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], return_dataset="tf")
    
    def _load_cache(self, cached_file):
        """
        Load a cached TFRecords dataset.

        Args:
          cached_file: the TFRecords file path.
        """
        if self.data_processor_name == "glue" or self.data_processor_name == "xnli":
            name_to_features = {
                "input_ids": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "attention_mask": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "token_type_ids": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "label": tf.io.FixedLenFeature([1], tf.int64),
            }
        elif self.data_processor_name.startswith("squad"):
            name_to_features = {
                "input_ids": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "attention_mask": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "token_type_ids": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "start_position": tf.io.FixedLenFeature([1], tf.int64),
                "end_position": tf.io.FixedLenFeature([1], tf.int64),
            }

        def _decode_record(record):
            if self.data_processor_name == "glue" or self.data_processor_name == "xnli":
                example = tf.io.parse_single_example(record, name_to_features)

                return {k:example[k] for k in ('input_ids','attention_mask','token_type_ids') if k in example}, example["label"]
            elif self.data_processor_name.startswith("squad"):
                example = tf.io.parse_single_example(record, name_to_features)
                
                return {k:example[k] for k in ('input_ids','attention_mask','token_type_ids') if k in example}, {k:example[k] for k in ('start_position','end_position') if k in example}

        d = tf.data.TFRecordDataset(cached_file)
        d = d.map(_decode_record, num_parallel_calls=4)

        return d
    
    def _save_cache(self, mode, cached_file):
        """
        Save a cached TFRecords dataset.

        Args:
          mode: the dataset to be cached.
          cached_file: the file path where the TFRecords will be saved.
        """
        writer = tf.io.TFRecordWriter(cached_file)

        for (feature, label) in self.datasets[mode]:
            def create_list_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f
            
            def create_int_feature(value):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
                return f

            record_feature = OrderedDict()
            record_feature["input_ids"] = create_list_int_feature(feature["input_ids"])
            record_feature["attention_mask"] = create_list_int_feature(feature["attention_mask"])
            record_feature["token_type_ids"] = create_list_int_feature(feature["token_type_ids"])
            
            if isinstance(label, dict):
                record_feature["start_position"] = create_int_feature(label["start_position"])
                record_feature["end_position"] = create_int_feature(label["end_position"])
            else:
                record_feature["label"] = create_int_feature(label)

            tf_example = tf.train.Example(features=tf.train.Features(feature=record_feature))

            writer.write(tf_example.SerializeToString())

        writer.close()
    
    def _preprocessing_data(self, tokenizer, data_dir, cache_dir):
        """
        Preprocess the training and validation data.

        Args:
          tokenizer: the tokenizer used for encoding the textual data into features.
          data_dir: the directory path where the data are.
          cache_dir: the directory path where the cached data are.
        """
        cached_training_features_file = os.path.join(
            cache_dir, "cached_train_{}_{}_{}.tf_record".format(self.data_processor_name,
                list(filter(None, self.pretrained_model_name_or_path.split("/"))).pop(), str(self.max_len)
            ),
        )
        cached_validation_features_file = os.path.join(
            cache_dir, "cached_validation_{}_{}_{}.tf_record".format(self.data_processor_name,
                list(filter(None, self.pretrained_model_name_or_path.split("/"))).pop(), str(self.max_len)
            ),
        )
        cached_test_features_file = os.path.join(
            cache_dir, "cached_test_{}_{}_{}.tf_record".format(self.data_processor_name,
                list(filter(None, self.pretrained_model_name_or_path.split("/"))).pop(), str(self.max_len)
            ),
        )

        if os.path.exists(cached_training_features_file) and os.path.exists(cached_validation_features_file):
            logger.info("Loading features from cached file %s", cached_training_features_file)
            self.datasets["train"] = self._load_cache(cached_training_features_file)
            logger.info("Loading features from cached file %s", cached_validation_features_file)
            self.datasets["validation"] = self._load_cache(cached_validation_features_file)
            logger.info("Loading features from cached file %s", cached_test_features_file)
            self.datasets["test"] = self._load_cache(cached_test_features_file)
        else:
            os.makedirs(cache_dir, exist_ok=True)
            self._create_features(tokenizer, data_dir)
            logger.info("Create cache file %s", cached_training_features_file)
            self._save_cache("train", cached_training_features_file)
            logger.info("Create cache file %s", cached_validation_features_file)
            self._save_cache("validation", cached_validation_features_file)
            logger.info("Create cache file %s", cached_test_features_file)
            self._save_cache("test", cached_test_features_file)
        
        self.train_steps = math.ceil(self.datasets["train"].reduce(0, lambda x, _: x + 1).numpy() / self.batch_size)
        self.datasets["train"] = self.datasets["train"].shuffle(128).batch(self.batch_size).repeat(-1)
        self.datasets["train"] = self.strategy.experimental_distribute_dataset(self.datasets["train"])
        self.validation_steps = math.ceil(self.datasets["validation"].reduce(0, lambda x, _: x + 1).numpy() / self.eval_batch_size)
        self.datasets["validation"] = self.datasets["validation"].batch(self.eval_batch_size)
        self.datasets["validation"] = self.strategy.experimental_distribute_dataset(self.datasets["validation"])
        self.test_steps = math.ceil(self.datasets["test"].reduce(0, lambda x, _: x + 1).numpy() / (self.eval_batch_size / self.strategy.num_replicas_in_sync))
        self.datasets["test"] = self.datasets["test"].batch(self.eval_batch_size // self.strategy.num_replicas_in_sync)

    def _create_optimizer(self):
        """
        Create the training optimizer with its name. Allowed names:
          - adam: Adam optimizer
          - adamw: Adam with Weight decay optimizer
          - adadelta: Adadelta optimizer
          - rmsprop: Root Mean Square Propogation optimizer
          - sgd: Stochastic Gradient Descent optimizer
        """
        if self.optimizer_name == "adamw":
            if self.decay_steps is None:
                self.decay_steps = self.train_steps
            
            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=self.learning_rate,
                                                                                decay_steps=self.decay_steps,
                                                                                end_learning_rate=0.0)
            if self.warmup_steps:
                learning_rate_fn = WarmUp(initial_learning_rate=self.learning_rate, decay_schedule_fn=learning_rate_fn,
                                            warmup_steps=self.warmup_steps)
            
            self.optimizer = AdamWeightDecay(learning_rate=learning_rate_fn, weight_decay_rate=0.01, epsilon=self.epsilon,
                                                exclude_from_weight_decay=["layer_norm", "bias"])
        elif self.optimizer_name == "adam":
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, epsilon=self.epsilon)
        elif self.optimizer_name == "adadelta":
            self.optimizer = tf.keras.optimizers.Adadelta(self.learning_rate, epsilon=self.epsilon)
        elif self.optimizer_name == "rms":
            self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate, epsilon=self.epsilon)
        elif self.optimizer_name == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        else:
            raise ValueError(
                "Allowed optimizers are adam, adamw, adadelta, rmsprop or sgd. Given: {}".format(self.optimizer_name)
            )

    def _create_checkpoint_manager(self, checkpoint_path, max_to_keep=5, load_model=True):
        """
        Create a checkpoint manager in order to be able to make the training
        fault-tolerant.

        Args:
          max_to_keep: the maximum number of checkpoints to keep in the
            checkpoint path.
          load_model: if we want to start the training from the latest checkpoint.
        """
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

        if load_model:
            ckpt.restore(self.model.ckpt_manager.latest_checkpoint)
    
    def _evaluate_during_training(self):
        num_batches = 0
        test_step = 1

        for batch in self.datasets["validation"]:
            test_step = tf.convert_to_tensor(test_step, dtype=tf.int64)
            self._distributed_test_step(batch)
            num_batches += 1
            
            with self.test_writer.as_default():
                tf.summary.scalar("loss", self.test_loss_metric.result(), step=test_step)
            
            if test_step % self.validation_steps == 0:
                break

            test_step += 1
    
    def train(self):
        """
        Train method to train the model.
        """
        with self.strategy.scope():
            tf.summary.trace_on(graph=True, profiler=True)
            
            step = 1
            train_loss = 0.0

            for epoch in range(1, self.epochs + 1):
                total_loss = 0.0
                num_batches = 0

                for batch in self.datasets["train"]:
                    step = tf.convert_to_tensor(step, dtype=tf.int64)
                    total_loss += self._distributed_train_step(batch)
                    num_batches += 1

                    with self.train_writer.as_default():
                        tf.summary.scalar("loss", total_loss / num_batches, step=step)
                    
                    if step == 1:
                        with self.train_writer.as_default():
                            tf.summary.trace_export(name="training", step=step, profiler_outdir=self.log_path)

                    if step % 10 == 0:
                        logger.info("Epoch {} Step {} Loss {:.4f} Train Accuracy {:.4f}".format(epoch, step.numpy(), total_loss.numpy() / num_batches, self.train_acc_metric.result()))

                    if step % 100 == 0:
                        ckpt_save_path = self.model.ckpt_manager.save()
                        logger.info("Saving checkpoint for step {} at {}".format(step, ckpt_save_path))
                    
                    if step % self.train_steps == 0:
                        step += 1
                        break

                    step += 1

                train_loss = total_loss / num_batches
                
                self._evaluate_during_training()

                logger.info("Epoch {} Step {} Train Loss {:.4f} Train Accuracy {:.4f}".format(epoch, step.numpy() - 1, train_loss.numpy(), self.train_acc_metric.result()))
                logger.info("Epoch {} Validation Loss {:.4f} Validation Accuracy {:.4f}".format(epoch, self.test_loss_metric.result(), self.test_acc_metric.result()))

            if epoch != self.epochs:
                self.train_acc_metric.reset_states()
                self.test_acc_metric.reset_states()

    @tf.function
    def _distributed_test_step(self, dist_inputs):
        """
        Method that represents a custom test step in distributed mode

        Args:
          dist_inputs: the features batch of the test data 
        """
        def step_fn(inputs):
            features, labels = inputs
            logits = self.model(features, training=False)

            if self.data_processor_name.startswith("squad"):
                loss = self.loss(labels, logits) + sum(self.model.losses)
            else:
                loss = self.loss(labels, logits[0]) + sum(self.model.losses)

            if self.data_processor_name.startswith("squad"):
                logger.info("Metric not available for SQuad")
            else:
                self.test_acc_metric(labels, logits)
            
            self.test_loss_metric(loss)

        self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
            
    @tf.function
    def _distributed_train_step(self, dist_inputs):
        """
        Method that represents a custom training step in distributed mode.

        Args:
          dist_inputs: the features batch of the training data
        """
        def step_fn(inputs):
            features, labels = inputs

            with tf.GradientTape() as tape:
                logits = self.model(features, training=True)
                
                if self.data_processor_name.startswith("squad"):
                    per_example_loss = self.loss(labels, logits)
                else:
                    per_example_loss = self.loss(labels, logits[0])
                
                loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)

            gradients = tape.gradient(loss, self.model.trainable_variables)

            if self.optimizer_name == "adamw":
                self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)), self.max_grad_norm)
            else:
                gradients = [(tf.clip_by_value(grad, -self.max_grad_norm, self.max_grad_norm)) for grad in gradients]
                self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))

            if self.data_processor_name.startswith("squad"):
                logger.info("Metric not available for SQuad")
            else:
                self.train_acc_metric(labels, logits)

            return loss

        per_replica_losses = self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
        sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        return sum_loss
    
    def evaluate(self):
        """
        Evaluate the model over the test dataset and print a report.
        """
        if not self.data_processor_name.startswith("squad"):
            y_true = []
            results = self.model.predict(self.datasets["test"], steps=self.test_steps)

            for batch in self.datasets["test"]:
                y_true.extend(batch[1].numpy().tolist())
            
            y_pred = np.reshape(np.argmax(results, axis=-1), (-1, 1)).tolist()
            y_true = list(itertools.chain.from_iterable(y_true))
            y_pred = list(itertools.chain.from_iterable(y_pred))

            logger.info(classification_report(y_true, y_pred, target_names=self.label2id.keys()))
        else:
            logger.info("Evaluation not yet available for SQuaD datasets")

    def save_model(self, save_path):
        """
        Save the pretrained model and create a Tensorflow saved model.

        Args:
          save_path: directory path where the pretrained model and
            Tensorflow saved model will be saved
        """
        logger.info("Saving model in {}".format(save_path))

        path = os.path.join(save_path, "saved_model")

        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(save_path)
        tf.saved_model.save(self.model, path)
    
