# coding=utf-8
""" Trainer class."""

from .optimization_tf import AdamWeightDecay, WarmUp
import tensorflow as tf
from .configuration_auto import AutoConfig
from .tokenization_auto import AutoTokenizer
from .modeling_tf_auto import TFAutoModel
from .data.processors.glue import glue_convert_examples_to_features, glue_processors

class Trainer(object):
    def __init__(self, **kwargs):
        self.pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        self.optimizer_name = kwargs.pop("optimizer_name", None)
        self.warmup_steps = kwargs.pop("warmup_steps", None)
        self.num_decay_steps = kwargs.pop("decay_steps", None)
        self.learning_rate = kwargs.pop("learning_rate", None)
        self.epsilon = kwargs.pop("epsilon", None)
        self.training_data = kwargs.pop("training_data", None)
        self.validation_data = kwargs.pop("validation_data", None)
        self.checkpoint_path = kwargs.pop("checkpoint_path", None)
        self.log_path = kwargs.pop("log_path", None)
        self.batch_size = kwargs.pop("batch_size", None)
        self.eval_batch_size = kwargs.pop("eval_batch_size", None)
        self.loss = kwargs.pop("loss", None)
        self.distributed = kwargs.pop("distributed", None)

        assert len(kwargs) == 0, "unrecognized params passed in: %s" % ",".join(kwargs.keys())
    
    def setup_training(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.preprocessing_data(tokenizer)
        config = AutoConfig.from_pretrained("bert-base-cased", num_labels=len(glue_processors["mrpc"]().get_labels()))
        self.model = TFAutoModel.from_pretrained("bert-base-cased", config=config)
        self.create_optimizer()
        self.create_checkpoint_manager()
        self.create_summary_writer()
    
    def create_summary_writer(self):
        """
        Create a summary writer to be able to read the logs in Tensorboard.
        """
        self.train_writer = tf.summary.create_file_writer(self.log_path + "/train")
        self.test_writer = tf.summary.create_file_writer(self.log_path + "/test")
    
    def preprocessing_data(self, tokenizer):
        """
        Preprocess the training and validation data.

        Args:
          tokenizer: the tokenizer used for encoding the textual data into features
        """
        self.training_data = glue_convert_examples_to_features(self.training_data, tokenizer, 128, "mrpc")
        self.training_data = self.training_data.shuffle(128).batch(self.batch_size).repeat(-1)

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
                                                                                decay_steps=self.num_decay_steps,
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

    def create_checkpoint_manager(self, max_to_keep=5, load_model=True):
        """
        Create a checkpoint manager in order to be able to make the training
        fault-tolerant.

        Args:
          max_to_keep: the maximum number of checkpoints to keep in the
            checkpoint path.
          load_model: if we want to start the training from the latest checkpoint.
        """
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=max_to_keep)

        if load_model:
            ckpt.restore(self.model.ckpt_manager.latest_checkpoint)
    
    def fit(self, train_dataset, **kwargs):
        """
        Fit method to train the model.

        Args:
          train_dataset: training dataset.
        """
        self.fit(train_dataset, **kwargs)
    
    def distributed_train_step(self, inputs, targets, step):
        """
        Method that represents a custom training step in distributed training mode.

        Args:
          inputs: the features batch of the training data
          targets: the labels batch of the training data
          step: training step number 
        """
        raise NotImplementedError

    def train_step(self, inputs, targets, step):
        """
        Method that represents a custom training step in single GPU training mode.

        Args:
          inputs: the features batch of the training data
          targets: the labels batch of the training data
          step: training step number
        """
        raise NotImplementedError