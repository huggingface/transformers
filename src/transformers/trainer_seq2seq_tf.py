from typing import Dict, Optional

import tensorflow as tf

from transformers import PreTrainedTokenizerBase, TFTrainer
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


class TFSeq2SeqTrainer(TFTrainer):
    """
    TFSeq2SeqTrainer is a simple but feature-complete training and eval loop for TensorFlow, optimized for 🤗
    Transformers.

    Args:
        tokenizer (:class:`PreTrainedTokenizerBase`, `optional`):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        max_length (:obj:`int`, `optional`):
            The maximum target length to use when predicting with the generate method.
        num_beams (:obj:`int`, `optional`):
            Number of beams for beam search that will be used when predicting with the generate method. 1 means no beam
            search.
    """

    def __init__(self, tokenizer: Optional["PreTrainedTokenizerBase"] = None, *args, **kwargs):
        super(TFSeq2SeqTrainer, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def evaluate(
        self,
        eval_dataset: Optional[tf.data.Dataset] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:class:`~tf.data.Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. The dataset should yield tuples of
                ``(features, labels)`` where ``features`` is a dict of input features and ``labels`` is the labels. If
                ``labels`` is a tensor, the loss is calculated by the model by calling ``model(features,
                labels=labels)``. If ``labels`` is a dict, such as when using a QuestionAnswering head model with
                multiple targets, the loss is instead calculated by calling ``model(features, **labels)``.
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        self._max_length = max_length
        self._num_beams = num_beams
        return super().evaluate(eval_dataset)

    def predict(
        self, test_dataset: tf.data.Dataset, max_length: Optional[int] = None, num_beams: Optional[int] = None
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:class:`~tf.data.Dataset`):
                Dataset to run the predictions on. The dataset should yield tuples of ``(features, labels)`` where
                ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the
                loss is calculated by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict,
                such as when using a QuestionAnswering head model with multiple targets, the loss is instead calculated
                by calling ``model(features, **labels)``
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        self._max_length = max_length
        self._num_beams = num_beams
        return super().predict(test_dataset)

    def distributed_prediction_steps(self, batch):
        nb_instances_in_batch = self._compute_nb_instances(batch)
        inputs = self._get_step_inputs(batch, nb_instances_in_batch)
        logits = self.args.strategy.run(self.prediction_step, inputs)

        return logits

    def prediction_step(
        self, features: tf.Tensor, labels: tf.Tensor, nb_instances_in_global_batch: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the prediction on features and update the loss with labels.

        Subclass and override to inject some custom behavior.
        """
        if not self.args.predict_with_generate:
            return super().prediction_step(features, labels, nb_instances_in_global_batch)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
        }

        generated_tokens = self.model.generate(
            features["input_ids"], attention_mask=features["attention_mask"], **gen_kwargs
        )

        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        return generated_tokens

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenizer was passed when creating "
                "this `TFTrainer`. Make sure to create your `TFTrainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        padded_tensor = tf.Variable(pad_token_id * tf.ones((tensor.shape[0], max_length), dtype=tensor.dtype))
        padded_tensor[:, : tensor.shape[-1]].assign(tensor)
        return padded_tensor
