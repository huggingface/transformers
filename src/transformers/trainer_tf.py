# coding=utf-8
""" Trainer class."""

import os
import logging
from collections import defaultdict

import tensorflow as tf
from .optimization_tf import *
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
        self.training_data = kwargs.pop("training_data", None)
        self.validation_data = kwargs.pop("validation_data", None)
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

        assert len(kwargs) == 0, "unrecognized params passed: %s" % ",".join(kwargs.keys())
    
    def setup_training(self, checkpoint_path, log_path, data_dir=None):
        if self.distributed:
            self.strategy = tf.distribute.MirroredStrategy()
            self.batch_size *= self.strategy.num_replicas_in_sync
        else:
            if tf.test.is_gpu_available():
                self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            else:
                self.strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        
        if self.data_processor_name == "glue":
            self.processor = glue_processors[self.task]
        elif self.data_processor_name == "squadv1":
            self.processor = SquadV1Processor()
        elif self.data_processor_name == "squadv2":
            self.processor = SquadV2Processor()
        elif self.data_processor_name == "xnli":
            self.processor = xnli_processors[self.task](language=self.language, train_language=self.train_language)

        self.preprocessing_data(tokenizer, data_dir)
            
        try:
            config = AutoConfig.from_pretrained(self.pretrained_model_name_or_path, num_labels=len(self.processor.get_labels()))
        except Exception as e:
            config = AutoConfig.from_pretrained(self.pretrained_model_name_or_path)

        with self.strategy.scope():
            self.model = globals()[self.architecture].from_pretrained(self.pretrained_model_name_or_path, config=config)
            self.create_optimizer()
            self.get_loss()
            self.create_checkpoint_manager(checkpoint_path)
            self.create_summary_writer(log_path)
    
    def get_loss(self):
        """
        Use the loss corresponding to the loss_name field.
        """
        if self.loss_name == "mean_squared_error":
            self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        elif self.loss_name == "sparse_categorical_crossentropy":
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
        
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    
    def create_summary_writer(self, log_path):
        """
        Create a summary writer to be able to read the logs in Tensorboard.
        """
        self.log_path = log_path
        self.train_writer = tf.summary.create_file_writer(log_path + "/train")
        self.test_writer = tf.summary.create_file_writer(log_path + "/test")
    
    def preprocessing_data(self, tokenizer, data_dir):
        """
        Preprocess the training and validation data.

        Args:
          tokenizer: the tokenizer used for encoding the textual data into features
        """
        if self.data_processor_name == "glue":
            self.training_data = glue_convert_examples_to_features(self.training_data, tokenizer, self.max_len, self.task)
            self.validation_data = glue_convert_examples_to_features(self.validation_data, tokenizer, self.max_len, self.task)
        elif self.data_processor_name == "squadv1" or self.data_processor_name == "squadv2":
            self.training_data = squad_convert_examples_to_features(self.training_data, tokenizer, self.max_len, self.doc_stride, self.max_query_len, True, return_dataset="tf")
            self.validation_data = squad_convert_examples_to_features(self.validation_data, tokenizer, self.max_len, self.doc_stride, self.max_query_len, False, return_dataset="tf")
        elif self.data_processor_name == "xnli":
            if data_dir is not None:
                def to_tensorflow_dataset(lst_features):
                    """Transform the examples into a Tensorflow Dataset"""
                    features_as_dict = [feat.to_dict() for feat in lst_features]
                    tf_features = defaultdict(list)
                    tf_labels = []

                    for d in features_as_dict:
                        for key, value in d.items():
                            if key == "label":
                                tf_labels.append(value)
                            else: 
                                tf_features[key].append(value)

                    tf_features = tf.data.Dataset.from_tensor_slices(dict(tf_features))
                    tf_labels = tf.data.Dataset.from_tensor_slices(tf_labels)

                    return tf.data.Dataset.zip((tf_features, tf_labels))

                train_examples = self.processor.get_train_examples(data_dir)
                self.training_data = glue_convert_examples_to_features(train_examples, tokenizer, max_length=self.max_len, label_list=self.processor.get_labels(), output_mode=xnli_output_modes[self.task], pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
                self.training_data = to_tensorflow_dataset(self.training_data)
                validation_examples = self.processor.get_test_examples(data_dir)
                self.validation_data = glue_convert_examples_to_features(validation_examples, tokenizer, max_length=self.max_len, label_list=self.processor.get_labels(), output_mode=xnli_output_modes[self.task], pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
                self.validation_data = to_tensorflow_dataset(self.validation_data)
            else:
                self.training_data = glue_convert_examples_to_features(self.training_data, tokenizer, max_length=self.max_len, label_list=self.processor.get_labels(), output_mode=xnli_output_modes[self.task], pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
                self.validation_data = glue_convert_examples_to_features(self.validation_data, tokenizer, max_length=self.max_len, label_list=self.processor.get_labels(), output_mode=xnli_output_modes[self.task], pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
        
        self.train_steps = self.training_data.reduce(0, lambda x, _: x + 1) // self.batch_size
        self.train_steps = self.train_steps.numpy()
        self.training_data = self.training_data.shuffle(128).batch(self.batch_size).repeat(-1)
        self.training_data = self.strategy.experimental_distribute_dataset(self.training_data)
        self.validation_steps = self.validation_data.reduce(0, lambda x, _: x + 1) // self.eval_batch_size
        self.validation_steps = self.validation_steps.numpy()
        self.validation_data = self.validation_data.batch(self.eval_batch_size)

    def create_optimizer(self):
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

    def create_checkpoint_manager(self, checkpoint_path, max_to_keep=5, load_model=True):
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

                for batch in self.training_data:
                    step = tf.convert_to_tensor(step, dtype=tf.int64)
                    total_loss += self.distributed_train_step(batch)
                    num_batches += 1

                    with self.train_writer.as_default():
                        tf.summary.scalar("loss", total_loss / num_batches, step=step)
                    
                    if step == 1:
                        with self.train_writer.as_default():
                            tf.summary.trace_export(name="training", step=step, profiler_outdir=self.log_path)

                    if step % 10 == 0:
                        print("Epoch {} Step {} Loss {:.4f} Train Accuracy {:.4f}".format(epoch, step.numpy(), total_loss.numpy() / num_batches, self.train_acc_metric.result()))
                        logger.info("Epoch {} Step {} Loss {:.4f} Train Accuracy {:.4f}".format(epoch, step.numpy(), total_loss.numpy(), self.train_acc_metric.result()))

                    if step % 100 == 0:
                        ckpt_save_path = self.model.ckpt_manager.save()
                        print("Saving checkpoint for step {} at {}".format(step, ckpt_save_path))
                        logger.info("Saving checkpoint for step {} at {}".format(step, ckpt_save_path))
                    
                    if step % self.train_steps == 0:
                        step += 1
                        break

                    step += 1

                train_loss = total_loss / num_batches
                num_batches = 0
                test_step = 1

                for batch in self.validation_data:
                    test_step = tf.convert_to_tensor(test_step, dtype=tf.int64)
                    self.distributed_test_step(batch)
                    num_batches += 1
                    
                    with self.test_writer.as_default():
                        tf.summary.scalar("loss", self.test_loss_metric.result(), step=test_step)
                    
                    if test_step % self.validation_steps == 0:
                        break

                    test_step += 1

                logger.info("Epoch {} Step {} Train Loss {:.4f} Train Accuracy {:.4f}".format(epoch, step.numpy() - 1, train_loss.numpy(), self.train_acc_metric.result()))
                print("Epoch {} Step {} Loss {:.4f}, Train Accuracy {:.4f}".format(epoch, step.numpy() - 1, train_loss.numpy(), self.train_acc_metric.result()))
                logger.info("Epoch {} Validation Loss {:.4f} Validation Accuracy {:.4f}".format(epoch, self.test_loss_metric.result(), self.test_acc_metric.result()))
                print("Epoch {} Validation Loss {:.4f} Validation Accuracy {:.4f}".format(epoch, self.test_loss_metric.result(), self.test_acc_metric.result()))

            if epoch != self.epochs:
                self.train_acc_metric.reset_states()
                self.test_acc_metric.reset_states()

    @tf.function
    def distributed_test_step(self, dist_inputs):
        """
        Method that represents a custom test step in distributed mode

        Args:
          dist_inputs: the features batch of the test data 
        """
        def step_fn(inputs):
            features, labels = inputs
            logits = self.model(features, training=False)[0]
            loss = self.loss(labels, logits) + sum(self.model.losses)

            self.test_acc_metric(labels, logits)
            self.test_loss_metric(loss)

        self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))

            
    @tf.function
    def distributed_train_step(self, dist_inputs):
        """
        Method that represents a custom training step in distributed mode.

        Args:
          dist_inputs: the features batch of the training data
        """
        def step_fn(inputs):
            features, labels = inputs

            with tf.GradientTape() as tape:
                logits = self.model(features, training=True)[0]
                per_example_loss = self.loss(labels, logits)
                loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            if self.optimizer_name == "adamw":
                self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)), self.max_grad_norm)
            else:
                gradients = [(tf.clip_by_value(grad, -self.max_grad_norm, self.max_grad_norm)) for grad in gradients]
                self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))

            self.train_acc_metric(labels, logits)

            return loss

        per_replica_losses = self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
        sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        return sum_loss

    def save_model(self, save_path):
        logger.info("Saving model in {}".format(save_path))

        path = os.path.join(save_path, "saved_model")

        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(save_path)
        tf.saved_model.save(self.model, path)
    
