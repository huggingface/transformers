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
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE * 2
USE_XLA = False
USE_AMP = False

def train_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.
    :return: ServeCommand
    """
    return TrainCommand(args.model)


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
                                  help='path to train (and optionally evaluation) dataset.')
        train_parser.add_argument('--task', type=str, default='text_classification',
                                  help='Task to train the model on.')
        train_parser.add_argument('--model', type=str, default='bert-base-uncased',
                                  help='Model\'s name or path to stored model.')
        train_parser.add_argument('--valid_data', type=str, default='',
                                  help='path to validation dataset.')
        train_parser.add_argument('--valid_data_ratio', type=float, default=0.1,
                                  help="if validation dataset is not provided, fraction of train dataset "
                                       "to use as validation dataset.")
        train_parser.set_defaults(func=train_command_factory)

    def __init__(self, model_name: str, task: str, train_data: str,
                 valid_data: str, valid_data_ratio: float):
        self._logger = getLogger('transformers-cli/training')

        self._framework = 'tf' if is_tf_available() else 'torch'

        self._logger.info('Loading model {}'.format(model_name))
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if task == 'text_classification':
            self._model = SequenceClassifModel.from_pretrained(model_name)
        elif task == 'token_classification':
            raise NotImplementedError
        elif task == 'question_answering':
            raise NotImplementedError

        dataset = SingleSentenceClassificationProcessor.create_from_csv(train_data)
        num_data_samples = len(SingleSentenceClassificationProcessor)
        if valid_data:
            self._train_dataset = dataset
            self._num_train_samples = num_data_samples
            self._valid_dataset = SingleSentenceClassificationProcessor.create_from_csv(valid_data)
            self._num_valid_samples = len(self._valid_dataset)
        else:
            assert 0.0 < valid_data_ratio < 1.0, "--valid_data_ratio should be between 0.0 and 1.0"
            self._num_valid_samples = num_data_samples * valid_data_ratio
            self._num_train_samples = num_data_samples - self._num_valid_samples
            self._train_dataset = dataset[self._num_train_samples]
            self._valid_dataset = dataset[self._num_valid_samples]

    def run(self):
        if self._framework == 'tf':
            return self.run_tf()
        return self.run_torch()

    def run_torch(self):
        raise NotImplementedError

    def run_tf(self):
        import tensorflow as tf

        tf.config.optimizer.set_jit(USE_XLA)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

        # Prepare dataset as a tf.train_data.Dataset instance
        train_dataset = convert_examples_to_features(self._train_dataset, self._tokenizer, mode='sequence_classification')
        valid_dataset = convert_examples_to_features(self._valid_dataset, self._tokenizer, mode='sequence_classification')
        train_dataset = train_dataset.shuffle(128).batch(BATCH_SIZE).repeat(-1)
        valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)

        # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule 
        opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
        if USE_AMP:
            # loss scaling is currently required when using mixed precision
            opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=opt, loss=loss, metrics=[metric])

        # Train and evaluate using tf.keras.Model.fit()
        train_steps = train_examples//BATCH_SIZE
        valid_steps = valid_examples//EVAL_BATCH_SIZE

        history = model.fit(train_dataset, epochs=2, steps_per_epoch=train_steps,
                            validation_data=valid_dataset, validation_steps=valid_steps)

        # Save TF2 model
        os.makedirs('./save/', exist_ok=True)
        model.save_pretrained('./save/')
