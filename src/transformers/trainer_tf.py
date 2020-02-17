# coding=utf-8
""" Trainer class."""

import os
import logging

from .optimization_tf import AdamWeightDecay, WarmUp
import tensorflow as tf
from .configuration_auto import AutoConfig
from .tokenization_auto import AutoTokenizer
from .modeling_tf_auto import TFAutoModelForSequenceClassification
from .data.processors.glue import glue_convert_examples_to_features, glue_processors


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

        assert len(kwargs) == 0, "unrecognized params passed: %s" % ",".join(kwargs.keys())
    
    def setup_training(self, checkpoint_path, log_path):
        if self.distributed:
            self.strategy = tf.distribute.MirroredStrategy()
            self.batch_size *= self.strategy.num_replicas_in_sync
        else:
            if tf.test.is_gpu_available():
                self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            else:
                self.strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.preprocessing_data(tokenizer)
        config = AutoConfig.from_pretrained(self.pretrained_model_name_or_path, num_labels=len(glue_processors["mrpc"]().get_labels()))

        with self.strategy.scope():
            self.model = TFAutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name_or_path, config=config)
            self.model.get_layer("classifier").activation = tf.keras.activations.softmax
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
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    def create_summary_writer(self, log_path):
        """
        Create a summary writer to be able to read the logs in Tensorboard.
        """
        self.log_path = log_path
        self.train_writer = tf.summary.create_file_writer(log_path + "/train")
        self.test_writer = tf.summary.create_file_writer(log_path + "/test")
    
    def preprocessing_data(self, tokenizer):
        """
        Preprocess the training and validation data.

        Args:
          tokenizer: the tokenizer used for encoding the textual data into features
        """
        self.training_data = glue_convert_examples_to_features(self.training_data, tokenizer, 128, "mrpc")
        self.train_steps = self.training_data.reduce(0, lambda x, _: x + 1) // self.batch_size
        self.train_steps = self.train_steps.numpy()
        self.training_data = self.training_data.shuffle(128).batch(self.batch_size).repeat(-1)
        self.training_data = self.strategy.experimental_distribute_dataset(self.training_data)
        self.validation_data = glue_convert_examples_to_features(self.validation_data, tokenizer, 128, "mrpc")
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
            if self.num_decay_steps is None:
                raise ValueError("The \"num_decay_steps\" parameter should not be None")
            
            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=self.learning_rate,
                                                                                decay_steps=self.decay_steps,
                                                                                end_learning_rate=0.0)
            if self.warmup_steps:
                learning_rate_fn = WarmUp(initial_learning_rate=self.learning_rate, decay_schedule_fn=learning_rate_fn,
                                            warmup_steps=self.warmup_steps)
            
            self.optimizer = AdamWeightDecay(learning_rate=learning_rate_fn, weight_decay_rate=0.01, epsilon=self.epsilon,
                                                exclude_from_weight_decay=["layer_norm", "bias"])
        if self.optimizer_name == "adam":
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
                    total_loss += self.distributed_train_step(batch, step)
                    num_batches += 1

                    with self.train_writer.as_default():
                        tf.summary.scalar("loss", total_loss / num_batches, step=step)
                    
                    if step == 1:
                        with self.train_writer.as_default():
                            tf.summary.trace_export(name="training", step=step, profiler_outdir=self.log_path)

                    if step % 10 == 0:
                        print("Epoch {} Step {} Loss {:.4f}".format(epoch, step.numpy(), total_loss.numpy() / num_batches))
                        logger.info("Epoch {} Step {} Loss {:.4f}".format(epoch, step.numpy(), total_loss.numpy()))

                    if step % 100 == 0:
                        ckpt_save_path = self.model.ckpt_manager.save()
                        logger.info("Save checkpoint")
                    
                    if step % self.train_steps == 0:
                        step += 1
                        break

                    step += 1

                train_loss = total_loss / num_batches

            logger.info("Epoch {} Step {} Loss {:.4f}".format(epoch, step.numpy() - 1, train_loss.numpy()))
            print("Epoch {} Step {} Loss {:.4f}".format(epoch, step.numpy() - 1, train_loss.numpy()))

            
    @tf.function
    def distributed_train_step(self, dist_inputs, step):
        """
        Method that represents a custom training step in distributed training mode.

        Args:
          dist_inputs: the features batch of the training data
          step: training step number 
        """
        def step_fn(inputs):
            features, labels = inputs

            with tf.GradientTape() as tape:
                logits = self.model(features, training=True)[0]
                cross_entropy = self.loss(labels, logits)
                loss = tf.nn.compute_average_loss(cross_entropy, global_batch_size=self.batch_size)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            gradients = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in gradients]

            self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))

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
    
