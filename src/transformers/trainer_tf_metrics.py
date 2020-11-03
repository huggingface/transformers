from typing import Optional

import tensorflow as tf


def accuracy_precision_recall_f1(
    confusion_matrix: tf.Tensor, num_classes: int, positive_indices: Optional[list] = None
):
    """
    Compute the accuracy, precision, recall and f1 scores from the confusion matrix

    Args:
        confusion_matrix (:obj:`tf.Tensor`):
            The confusion matrix.
        num_classes (:obj:`int`):
            The number of unique classes in the dataset.
        positive_indices (:obj:`list`, `optional`, defaults to None):
            The indices of the positive classe

    Returns:
        A tuple (accuracy, precision, recall, f1)
    """
    mask = [True if i in positive_indices else False for i in range(num_classes)]
    confusion_matrix_mask = tf.ones([num_classes])
    confusion_matrix_mask = tf.where(mask, confusion_matrix_mask, 0)
    diag_sum = tf.reduce_sum(tf.linalg.diag_part(confusion_matrix * confusion_matrix_mask))
    confusion_matrix_mask = tf.ones([num_classes, num_classes])
    confusion_matrix_mask = tf.where(mask, confusion_matrix_mask, 0)
    tot_pred = tf.reduce_sum(confusion_matrix * confusion_matrix_mask)
    mask = [[True] if i in positive_indices else [False] for i in range(num_classes)]
    confusion_matrix_mask = tf.ones([num_classes, num_classes])
    confusion_matrix_mask = tf.where(mask, confusion_matrix_mask, 0)
    tot_gold = tf.reduce_sum(confusion_matrix * confusion_matrix_mask)
    accuracy_score = tf.math.divide_no_nan(
        tf.reduce_sum(tf.linalg.diag_part(confusion_matrix)), tf.reduce_sum(confusion_matrix)
    )
    precison_score = tf.math.divide_no_nan(diag_sum, tot_pred)
    recall_score = tf.math.divide_no_nan(diag_sum, tot_gold)
    f1_score = tf.math.divide_no_nan(2 * (precison_score * recall_score), (precison_score + recall_score))

    return accuracy_score, precison_score, recall_score, f1_score


def metrics_from_confusion_matrix(
    confusion_matrix: tf.Tensor, num_classes: int, positive_indices: Optional[list] = None, average: str = "micro"
):
    """
    Compute the Precision, Recall and F1 scores from the confusion matri

    Args:
        confusion_matrix (:obj:`tf.Tensor`):
            The confusion matrix.
        num_classes (:obj:`int`):
            The number of unique classes in the dataset.
        pos_indices (:obj:`list`, `optional`, defaults to None):
            The indices of the positive classes, default is all
        average (:obj:`str`, `optional`, defaults to micro):
            micro: True positivies, false positives and false negatives are computed globally for the classes in
            `pos_indices`. 'macro': True positivies, false positives and false negatives are computed for each class in
            `pos_indices` and their unweighted mean is returned

    Returns:
        A tuple (accuracy, precision, recall, f1)
    """
    if positive_indices is None:
        positive_indices = [i for i in range(num_classes)]

    if average == "micro":
        return accuracy_precision_recall_f1(confusion_matrix, num_classes, positive_indices)
    else:
        precisions = []
        recalls = []
        f1s = []
        golds = []
        accuracies = []

        for idx in positive_indices:
            accuracy_score, precision_score, recall_score, f1_score = accuracy_precision_recall_f1(
                confusion_matrix, num_classes, [idx]
            )

            accuracies.append(accuracy_score)
            precisions.append(precision_score)
            recalls.append(recall_score)
            f1s.append(f1_score)

            mask = [[False] if i in [idx] else [True] for i in range(num_classes)]
            confusion_matrix_mask = tf.zeros([num_classes, num_classes])
            confusion_matrix_mask = tf.where(mask, confusion_matrix_mask, 1)

            golds.append(tf.reduce_sum(confusion_matrix * confusion_matrix_mask))

        accuracy_score = tf.reduce_mean(accuracies)
        precision_score = tf.reduce_mean(precisions)
        recall_score = tf.reduce_mean(recalls)
        f1_score = tf.reduce_mean(f1s)

        return accuracy_score, precision_score, recall_score, f1_score


class F1AndAccuracyMeanScore(tf.keras.metrics.Metric):
    """
    Computes the mean between the F-1 and the accuracy scores. F-1 is the weighted harmonic mean of precision and
    recall. Works for both multi-class and multi-label classification. F-1 = 2 * ((prec * recall) / (prec + recall))
    Accuracy is how often the predictions equals the labels

    Args:
        num_classes (:obj:`int`):
            The number of unique classes in the dataset.
        positive_indices (:obj:`list`, `optional`, defaults to None):
            The indices of the positive classes, default is all
        average (:obj:`str`, `optional`, defaults to micro):
            micro: True positivies, false positives and false negatives are computed globally for the classes in
            `pos_indices`. macro: True positivies, false positives and false negatives are computed for each class in
            `pos_indices` and their unweighted mean is returned

    Returns:
        The mean between the F1 and the accuracy scores

    Raises:
        ValueError: If the `average` has values other than [micro, macro].
    """

    def __init__(
        self,
        num_classes,
        positive_indices=None,
        average="micro",
        name="f1_acc_score",
    ):
        super().__init__(name=name)

        if average not in (None, "micro", "macro"):
            raise ValueError("Unknown average type. Acceptable values " "are: [micro, macro]")

        self.num_classes = num_classes
        self.average = average
        self.positive_indices = positive_indices
        self.confusion_matrices = self.add_weight(
            "confusion_matrices", shape=(num_classes, num_classes), initializer="zeros", dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, 1)
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)

        self.confusion_matrices.assign_add(confusion_matrix)

    def result(self):
        accuracy_score, _, _, f1_score = metrics_from_confusion_matrix(
            self.confusion_matrices, self.num_classes, self.positive_indices, self.average
        )

        acc_and_f1 = (accuracy_score + f1_score) / 2

        return acc_and_f1

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "positive_indices": self.positive_indices,
        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.confusion_matrices.assign(tf.zeros((self.num_classes, self.num_classes), tf.float32))


class F1Score(tf.keras.metrics.Metric):
    """
    Computes the F-1 score

    Args:
        num_classes (:obj:`int`):
            The number of unique classes in the dataset.
        positive_indices (:obj:`list`, `optional`, defaults to None):
            The indices of the positive classes, default is all
        average (:obj:`str`, `optional`, defaults to micro):
            micro: True positivies, false positives and false negatives are computed globally for the classes in
            `pos_indices`. macro: True positivies, false positives and false negatives are computed for each class in
            `pos_indices` and their unweighted mean is returned

    Returns:
        The F1 score

    Raises:
        ValueError: If the `average` has values other than [micro, macro].
    """

    def __init__(
        self,
        num_classes,
        positive_indices=None,
        average="micro",
        name="f1_score",
    ):
        super().__init__(name=name)

        if average not in ("micro", "macro"):
            raise ValueError("Unknown average type. Acceptable values " "are: [micro, macro]")

        self.num_classes = num_classes
        self.average = average
        self.positive_indices = positive_indices
        self.confusion_matrices = self.add_weight(
            "confusion_matrices", shape=(num_classes, num_classes), initializer="zeros", dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, 1)
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)

        self.confusion_matrices.assign_add(confusion_matrix)

    def result(self):
        _, _, _, f1_score = metrics_from_confusion_matrix(
            self.confusion_matrices, self.num_classes, self.positive_indices, self.average
        )

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "positive_indices": self.positive_indices,
        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.confusion_matrices.assign(tf.zeros((self.num_classes, self.num_classes), tf.float32))
