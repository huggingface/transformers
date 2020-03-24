# coding=utf-8
""" Trainer class."""

import os
import logging
from collections import OrderedDict
import math
import itertools
from abc import ABC, abstractmethod

import numpy as np

from sklearn.metrics import classification_report

import tensorflow as tf
from .optimization_tf import WarmUp, AdamWeightDecay
from .losses_tf import QALoss
from .configuration_auto import AutoConfig
from .tokenization_auto import AutoTokenizer
from .data.processors.task_processors import DataProcessorForSequenceClassification
from .modeling_tf_auto import TFAutoModelForSequenceClassification


logger = logging.getLogger(__name__)


class TFTrainer(ABC):
    def __init__(self, **kwargs):
        """
        The list of keys in kwargs here should be generic to all the possible models/architectures
        and not specific to such or such dataset/task.
        """
        self.pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        self.optimizer_name = kwargs.pop("optimizer_name", None)
        self.warmup_steps = kwargs.pop("warmup_steps", None)
        self.decay_steps = kwargs.pop("decay_steps", None)
        self.learning_rate = kwargs.pop("learning_rate", None)
        self.adam_epsilon = kwargs.pop("adam_epsilon", 1e-08)
        self.loss_name = kwargs.pop("loss_name", None)
        self.batch_size = kwargs.pop("batch_size", None)
        self.eval_batch_size = kwargs.pop("eval_batch_size", None)
        self.distributed = kwargs.pop("distributed", None)
        self.epochs = kwargs.pop("epochs", None)
        self.max_grad_norm = kwargs.pop("max_grad_norm", 1.0)
        self.metric_name = kwargs.pop("metric_name", None)
        self.max_len = kwargs.pop("max_len", None)
        self.task = kwargs.pop("task", None)
        self.datasets = {}
        self.processor = None
        self.model_class = None

        if self.distributed:
            self.strategy = tf.distribute.MirroredStrategy()
            self.batch_size *= self.strategy.num_replicas_in_sync
            self.eval_batch_size *= self.strategy.num_replicas_in_sync
        else:
            if len(tf.config.list_physical_devices('GPU')) >= 1:
                self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            else:
                self.strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

        # assert len(kwargs) == 0, "unrecognized params passed: %s" % ",".join(kwargs.keys())

    def setup_training(self, checkpoint_path="checkpoints", log_path="logs", data_cache_dir="cache", model_cache_dir=None):
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
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path, cache_dir=model_cache_dir, use_fast=False)

        self._preprocess_data(data_cache_dir)
        self._config_trainer()

        with self.strategy.scope():
            self.model = self.model_class.from_pretrained(self.pretrained_model_name_or_path, config=self.config, cache_dir=model_cache_dir)
            self._create_optimizer()
            self._set_loss_and_metric()
            self._create_checkpoint_manager(checkpoint_path)
            self._create_summary_writer(log_path)

    def _config_trainer(self, model_cache_dir):
        """
        This method set all the required fields for a specific task. For example
        in case of a classification set all the labels.
        """
        self.config = AutoConfig.from_pretrained(self.pretrained_model_name_or_path, cache_dir=model_cache_dir)

    def _set_loss_and_metric(self):
        """
        Use the loss corresponding to the loss_name field.
        """
        if self.loss_name == "qa_loss":
            self.loss = QALoss()
        else:
            try:
                self.loss = tf.keras.losses.get({"class_name": self.loss_name, "config": {"from_logits": True, "reduction": tf.keras.losses.Reduction.NONE}})
            except TypeError:
                self.loss = tf.keras.losses.get({"class_name": self.loss_name, "config": {"reduction": tf.keras.losses.Reduction.NONE}})

        if self.metric_name == "qa_metric":
            pass
        else:
            self.train_acc_metric = tf.keras.metrics.get({"class_name": self.metric_name, "config": {"name": "train_accuracy"}})
            self.test_acc_metric = tf.keras.metrics.get({"class_name": self.metric_name, "config": {"name": "test_accuracy"}})

        self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

    def _create_summary_writer(self, log_path):
        """
        Create a summary writer to be able to read the logs in Tensorboard.
        """
        self.log_path = log_path
        self.train_writer = tf.summary.create_file_writer(log_path + "/train")
        self.test_writer = tf.summary.create_file_writer(log_path + "/test")

    @abstractmethod
    def _create_features(self):
        """
        Create the features for the training and validation data.
        """
        pass

    @abstractmethod
    def _load_cache(self, cached_file):
        """
        Load a cached TFRecords dataset.
        Args:
          cached_file: the TFRecords file path.
        """
        pass

    @abstractmethod
    def _save_cache(self, mode, cached_file):
        """
        Save a cached TFRecords dataset.
        Args:
          mode: the dataset to be cached.
          cached_file: the file path where the TFRecords will be saved.
        """
        pass

    def _preprocess_data(self, cache_dir):
        """
        Preprocess the training and validation data.
        Args:
          tokenizer: the tokenizer used for encoding the textual data into features.
          data_dir: the directory path where the data are.
          cache_dir: the directory path where the cached data are.
        """
        cached_training_features_file = os.path.join(
            cache_dir, "cached_train_{}_{}_{}.tf_record".format(
                self.task, list(filter(None, self.pretrained_model_name_or_path.split("/"))).pop(), str(self.max_len)
            ),
        )
        cached_validation_features_file = os.path.join(
            cache_dir, "cached_validation_{}_{}_{}.tf_record".format(
                self.task, list(filter(None, self.pretrained_model_name_or_path.split("/"))).pop(), str(self.max_len)
            ),
        )
        cached_test_features_file = os.path.join(
            cache_dir, "cached_test_{}_{}_{}.tf_record".format(
                self.task, list(filter(None, self.pretrained_model_name_or_path.split("/"))).pop(), str(self.max_len)
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
            self.processor.create_examples()
            self._create_features()
            logger.info("Create cache file %s", cached_training_features_file)
            self._save_cache("train", cached_training_features_file)
            logger.info("Create cache file %s", cached_validation_features_file)
            self._save_cache("validation", cached_validation_features_file)
            logger.info("Create cache file %s", cached_test_features_file)
            self._save_cache("test", cached_test_features_file)

        self.train_steps = math.ceil(self.processor.num_examples("train") / self.batch_size)
        self.datasets["train"] = self.datasets["train"].shuffle(128).batch(self.batch_size).repeat(-1)
        self.datasets["train"] = self.strategy.experimental_distribute_dataset(self.datasets["train"])
        self.validation_steps = math.ceil(self.processor.num_examples("validation") / self.eval_batch_size)
        self.datasets["validation"] = self.datasets["validation"].batch(self.eval_batch_size)
        self.datasets["validation"] = self.strategy.experimental_distribute_dataset(self.datasets["validation"])
        self.test_steps = math.ceil(self.processor.num_examples("test") / (self.eval_batch_size / self.strategy.num_replicas_in_sync))
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

            self.optimizer = AdamWeightDecay(learning_rate=learning_rate_fn, weight_decay_rate=0.01, epsilon=self.adam_epsilon,
                                             exclude_from_weight_decay=["layer_norm", "bias"])
        else:
            try:
                self.optimizer = tf.keras.optimizers.get({"class_name": self.optimizer_name, "config" : {"learning_rate": self.learning_rate, "epsilon": self.adam_epsilon}})
            except TypeError:
                # This is for the case where the optimizer is not Adam-like such as SGD
                self.optimizer = tf.keras.optimizers.get({"class_name": self.optimizer_name, "config" : {"learning_rate": self.learning_rate}})

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
        """
        Evaluate the model during the training at the end of each epoch.
        """
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

    @abstractmethod
    def _distributed_test_step(self, dist_inputs):
        """
        Method that represents a custom test step in distributed mode
        Args:
          dist_inputs: the features batch of the test data
        """
        pass

    @abstractmethod
    def _distributed_train_step(self, dist_inputs):
        """
        Method that represents a custom training step in distributed mode.
        Args:
          dist_inputs: the features batch of the training data
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluate the model over the test dataset and print a report.
        """
        pass

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


class TFTrainerForSequenceClassification(TFTrainer):
    def __init__(self, **config):
        model_config = config.pop("model_config", None)
        data_processor_config = config.pop("data_processor_config", None)

        if model_config is None or data_processor_config is None:
            raise ValueError("the model_config and data_processor_config properties should not be empty from the configuration")

        super().__init__(**model_config)
        self.processor = DataProcessorForSequenceClassification(**data_processor_config)
        self.model_class = TFAutoModelForSequenceClassification
        self.labels = []

    def _create_features(self):
        self.datasets["train"] = self.processor.convert_examples_to_features("train", self.tokenizer, self.max_len, return_dataset="tf")
        self.datasets["validation"] = self.processor.convert_examples_to_features("validation", self.tokenizer, self.max_len, return_dataset="tf")
        self.datasets["test"] = self.processor.convert_examples_to_features("test", self.tokenizer, self.max_len, return_dataset="tf")

        if self.datasets["test"] is None:
            self.datasets["test"] = self.datasets["validation"]

    def get_labels(self):
        """
        Returns the list of labels associated to the trained model.
        """
        return self.labels

    def _config_trainer(self, model_cache_dir):
        self.labels = self.processor.get_labels()
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.config = AutoConfig.from_pretrained(self.pretrained_model_name_or_path, num_labels=len(self.labels), id2label=self.id2label, label2id=self.label2id, cache_dir=model_cache_dir)

    def _load_cache(self, cached_file):
        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([self.max_len], tf.int64),
            "attention_mask": tf.io.FixedLenFeature([self.max_len], tf.int64),
            "token_type_ids": tf.io.FixedLenFeature([self.max_len], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        }

        def _decode_record(record):
            example = tf.io.parse_single_example(record, name_to_features)

            return {k : example[k] for k in ('input_ids', 'attention_mask', 'token_type_ids') if k in example}, example["label"]

        d = tf.data.TFRecordDataset(cached_file)
        d = d.map(_decode_record, num_parallel_calls=4)

        return d

    def _save_cache(self, mode, cached_file):
        writer = tf.io.TFRecordWriter(cached_file)
        ds = self.datasets[mode].enumerate()

        # as_numpy_iterator() is available since TF 2.1
        for (index, (feature, label)) in ds.as_numpy_iterator():
            if index % 10000 == 0:
                logger.info("Writing example %d", index)

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
            record_feature["label"] = create_int_feature(label)
            tf_example = tf.train.Example(features=tf.train.Features(feature=record_feature))

            writer.write(tf_example.SerializeToString())

        writer.close()

    @tf.function
    def _distributed_test_step(self, dist_inputs):
        def step_fn(inputs):
            features, labels = inputs
            logits = self.model(features, training=False)
            loss = self.loss(labels, logits[0]) + sum(self.model.losses)

            self.test_acc_metric(labels, logits)
            self.test_loss_metric(loss)

        self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))

    @tf.function
    def _distributed_train_step(self, dist_inputs):
        def step_fn(inputs):
            features, labels = inputs

            with tf.GradientTape() as tape:
                logits = self.model(features, training=True)
                per_example_loss = self.loss(labels, logits[0])
                loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)

            gradients = tape.gradient(loss, self.model.trainable_variables)

            if self.optimizer_name == "adamw":
                self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)), self.max_grad_norm)
            else:
                gradients = [(tf.clip_by_value(grad, -self.max_grad_norm, self.max_grad_norm)) for grad in gradients]
                self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))

            self.train_acc_metric(labels, logits[0])

            return loss

        per_replica_losses = self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
        sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        return sum_loss

    def evaluate(self):
        y_true = []
        results = self.model.predict(self.datasets["test"], steps=self.test_steps)

        for batch in self.datasets["test"]:
            y_true.extend(batch[1].numpy().tolist())

        y_pred = np.reshape(np.argmax(results, axis=-1), (-1, 1)).tolist()
        y_true = list(itertools.chain.from_iterable(y_true))
        y_pred = list(itertools.chain.from_iterable(y_pred))

        logger.info(classification_report(y_true, y_pred, target_names=self.label2id.keys()))
