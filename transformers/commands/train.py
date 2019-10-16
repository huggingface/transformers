import os
from argparse import ArgumentParser, Namespace
from logging import getLogger

from transformers.commands import BaseTransformersCLICommand
from transformers import (AutoTokenizer, is_tf_available, is_torch_available,
                          SingleSentenceClassificationProcessor,
                          convert_examples_to_features)
if is_tf_available():
    from transformers import TFAutoModelForSequenceClassification as SequenceClassifModel
elif is_torch_available():
    from transformers import AutoModelForSequenceClassification as SequenceClassifModel
else:
    raise ImportError("At least one of PyTorch or TensorFlow 2.0+ should be installed to use CLI training")

# TF training parameters
USE_XLA = False
USE_AMP = False

def train_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.
    :return: ServeCommand
    """
    return TrainCommand(args)


class TrainCommand(BaseTransformersCLICommand):

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli
        :param parser: Root parser to register command-specific arguments
        :return:
        """
        train_parser = parser.add_parser('train', help='CLI tool to train a model on a task.')
        train_parser.add_argument('--train_data', type=str, required=True,
                                  help="path to train (and optionally evaluation) dataset as a csv with "
                                       "tab separated labels and sentences.")

        train_parser.add_argument('--column_label', type=int, default=0,
                                  help='Column of the dataset csv file with example labels.')
        train_parser.add_argument('--column_text', type=int, default=1,
                                  help='Column of the dataset csv file with example texts.')
        train_parser.add_argument('--column_id', type=int, default=2,
                                  help='Column of the dataset csv file with example ids.')

        train_parser.add_argument('--validation_data', type=str, default='',
                                  help='path to validation dataset.')
        train_parser.add_argument('--validation_split', type=float, default=0.1,
                                  help="if validation dataset is not provided, fraction of train dataset "
                                       "to use as validation dataset.")

        train_parser.add_argument('--output', type=str, default='./',
                                  help='path to saved the trained model.')

        train_parser.add_argument('--task', type=str, default='text_classification',
                                  help='Task to train the model on.')
        train_parser.add_argument('--model', type=str, default='bert-base-uncased',
                                  help='Model\'s name or path to stored model.')
        train_parser.add_argument('--train_batch_size', type=int, default=32,
                                  help='Batch size for training.')
        train_parser.add_argument('--valid_batch_size', type=int, default=64,
                                  help='Batch size for validation.')
        train_parser.add_argument('--learning_rate', type=float, default=3e-5,
                                  help="Learning rate.")
        train_parser.add_argument('--adam_epsilon', type=float, default=1e-08,
                                  help="Epsilon for Adam optimizer.")
        train_parser.set_defaults(func=train_command_factory)

    def __init__(self, args: Namespace):
        self.logger = getLogger('transformers-cli/training')

        self.framework = 'tf' if is_tf_available() else 'torch'

        os.makedirs(args.output)
        self.output = args.output

        self.column_label = args.column_label
        self.column_text = args.column_text
        self.column_id = args.column_id

        self.logger.info('Loading model {}'.format(args.model_name))
        self.model_name = args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if args.task == 'text_classification':
            self.model = SequenceClassifModel.from_pretrained(args.model_name)
        elif args.task == 'token_classification':
            raise NotImplementedError
        elif args.task == 'question_answering':
            raise NotImplementedError

        self.logger.info('Loading dataset from {}'.format(args.train_data))
        dataset = SingleSentenceClassificationProcessor.create_from_csv(args.train_data)
        num_data_samples = len(dataset)
        if args.validation_data:
            self.logger.info('Loading validation dataset from {}'.format(args.validation_data))
            self.valid_dataset = SingleSentenceClassificationProcessor.create_from_csv(args.validation_data)
            self.num_valid_samples = len(self.valid_dataset)
            self.train_dataset = dataset
            self.num_train_samples = num_data_samples
        else:
            assert 0.0 < args.validation_split < 1.0, "--validation_split should be between 0.0 and 1.0"
            self.num_valid_samples = num_data_samples * args.validation_split
            self.num_train_samples = num_data_samples - self.num_valid_samples
            self.train_dataset = dataset[self.num_train_samples]
            self.valid_dataset = dataset[self.num_valid_samples]

        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.learning_rate = args.learning_rate
        self.adam_epsilon = args.adam_epsilon

    def run(self):
        if self.framework == 'tf':
            return self.run_tf()
        return self.run_torch()

    def run_torch(self):
        raise NotImplementedError

    def run_tf(self):
        import tensorflow as tf

        tf.config.optimizer.set_jit(USE_XLA)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

        # Prepare dataset as a tf.train_data.Dataset instance
        self.logger.info('Tokenizing and processing dataset')
        train_dataset = self.train_dataset.get_features(self.tokenizer)
        valid_dataset = self.valid_dataset.get_features(self.tokenizer)
        train_dataset = train_dataset.shuffle(128).batch(self.train_batch_size).repeat(-1)
        valid_dataset = valid_dataset.batch(self.valid_batch_size)

        # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule 
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, epsilon=self.adam_epsilon)
        if USE_AMP:
            # loss scaling is currently required when using mixed precision
            opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=opt, loss=loss, metrics=[metric])

        # Train and evaluate using tf.keras.Model.fit()
        train_steps = self.num_train_samples//self.train_batch_size
        valid_steps = self.num_valid_samples//self.valid_batch_size

        self.logger.info('Training model')
        history = self.model.fit(train_dataset, epochs=2, steps_per_epoch=train_steps,
                                 validation_data=valid_dataset, validation_steps=valid_steps)

        # Save trained model
        self.model.save_pretrained(self.output)
