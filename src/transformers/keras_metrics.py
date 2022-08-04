import tensorflow as tf
import numpy as np


class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, label_to_ignore=-100):
        self.label_to_ignore = label_to_ignore
        self.correct_predictions = self.add_weight(name='correct_predictions', initializer='zeros', dtype=tf.int64)
        self.all_predictions = self.add_weight(name='all_predictions', initializer='zeros', dtype=tf.int64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        class_predictions = tf.math.argmax(y_pred, axis=-1)
        self.correct_predictions += tf.math.count_nonzero((class_predictions == y_true) & (y_true != label_to_ignore))
        self.all_predictions += tf.math.count_nonzero(y_true != label_to_ignore)

    def result(self):
        return tf.cast(self.correct_predictions, tf.float32) / tf.cast(self.all_predictions, tf.float32)


class MaskedMultiClassPrecision(tf.keras.metrics.Metric):
    def __init__(self, label_id_to_name, label_to_ignore=-100):
        if sorted(label_id_to_name.keys()) != list(range(max(label_id_to_name.keys()))):
            raise ValueError("label_id_to_name should be a dict whose keys are sequential integers from 0!")
        self.num_classes = len(label_id_to_name)
        self.label_id_to_name = label_id_to_name
        self.label_to_ignore = label_to_ignore
        self.tp = self.add_weight(shape=(self.num_classes,), name="tp",
                                              initializer='zeros', dtype=tf.int64)
        self.tp_plus_fp = self.add_weight(shape=(self.num_classes,), name="tp_plus_fp",
                                              initializer='zeros', dtype=tf.int64)

    def update_state(self, y_true, y_pred):
        class_predictions = tf.math.argmax(y_pred, axis=-1)
        true_positive_weights_mask = tf.cast(class_predictions == y_true, dtype=tf.int64)
        self.tp_plus_fp += tf.math.bincount(class_predictions, minlength=self.num_classes, maxlength=self.num_classes)
        self.tp += tf.math.bincount(class_predictions, minlength=self.num_classes, maxlength=self.num_classes,
                                                weights=true_positive_weights_mask)

    def result(self):
        precisions = tf.cast(self.tp, tf.float32) / tf.cast(tf.clip_by_value(self.tp_plus_fp, 1, 1e12), tf.float32)
        return {f"{class_name}_precision": precisions[i] for i, class_name in self.label_id_to_name.items()}


class MaskedMultiClassRecall(tf.keras.metrics.Metric):
    def __init__(self, label_id_to_name, label_to_ignore=-100):
        if sorted(label_id_to_name.keys()) != list(range(max(label_id_to_name.keys()))):
            raise ValueError("label_id_to_name should be a dict whose keys are sequential integers from 0!")
        self.num_classes = len(label_id_to_name)
        self.label_id_to_name = label_id_to_name
        self.label_to_ignore = label_to_ignore
        self.tp = self.add_weight(shape=(self.num_classes,), name="tp",
                                              initializer='zeros', dtype=tf.int64)
        self.tp_plus_fn = self.add_weight(shape=(self.num_classes,), name="tp_plus_fn",
                                                      initializer='zeros', dtype=tf.int64)

    def update_state(self, y_true, y_pred):
        # Mask with a too-high value that will be ignored by bincount
        y_true = tf.where(y_true != self.label_to_ignore, y_true, self.num_classes)
        true_positive_weights_mask = tf.cast(tf.math.argmax(y_pred, axis=-1) == y_true, dtype=tf.int64)
        self.tp_plus_fn += tf.math.bincount(y_true, minlength=self.num_classes, maxlength=self.num_classes)
        self.tp += tf.math.bincount(y_true, minlength=self.num_classes, maxlength=self.num_classes,
                                                weights=true_positive_weights_mask)

    def result(self):
        recalls = tf.cast(self.tp, tf.float32) / tf.cast(tf.clip_by_value(self.tp_plus_fn, 1, 1e12), tf.float32)
        return {f"{class_name}_recall": recalls[i] for i, class_name in self.label_id_to_name.items()}

class MaskedMultiClassF1(tf.keras.metrics.Metric):
    def __init__(self, label_id_to_name, label_to_ignore=-100):
        if sorted(label_id_to_name.keys()) != list(range(max(label_id_to_name.keys()))):
            raise ValueError("label_id_to_name should be a dict whose keys are sequential integers from 0!")
        self.num_classes = len(label_id_to_name)
        self.label_id_to_name = label_id_to_name
        self.label_to_ignore = label_to_ignore
        self.tp = self.add_weight(shape=(self.num_classes,), name="tp",
                                  initializer='zeros', dtype=tf.int64)
        self.tp_plus_fn = self.add_weight(shape=(self.num_classes,), name="tp_plus_fn",
                                          initializer='zeros', dtype=tf.int64)
        self.tp_plus_fp = self.add_weight(shape=(self.num_classes,), name="tp_plus_fp",
                                          initializer='zeros', dtype=tf.int64)

    def update_state(self, y_true, y_pred):
        class_predictions = tf.math.argmax(y_pred, axis=-1)
        y_true = tf.where(y_true != self.label_to_ignore, y_true, self.num_classes)
        true_positive_weights_mask = tf.cast(class_predictions == y_true, dtype=tf.int64)
        self.tp_plus_fp += tf.math.bincount(class_predictions, minlength=self.num_classes, maxlength=self.num_classes)
        self.tp += tf.math.bincount(class_predictions, minlength=self.num_classes, maxlength=self.num_classes,
                                                weights=true_positive_weights_mask)
        self.tp_plus_fn += tf.math.bincount(y_true, minlength=self.num_classes, maxlength=self.num_classes)

    def result(self):
        precisions = tf.cast(self.tp, tf.float32) / tf.cast(tf.clip_by_value(self.tp_plus_fp, 1, 1e12), tf.float32)
        recalls = tf.cast(self.tp, tf.float32) / tf.cast(tf.clip_by_value(self.tp_plus_fn, 1, 1e12), tf.float32)
        f1_scores = (2 * precisions * recalls) / tf.clip_by_value(precisions + recalls, 1., float(1e12))
        return {f"{class_name}_f1": f1_scores[i] for i, class_name in self.label_id_to_name.items()}


class MaskedBinaryF1(tf.keras.metrics.Metric):
    def __init__(self, label_to_ignore=-100):
        self.label_to_ignore = label_to_ignore
        self.tp = self.add_weight(name="tp", initializer='zeros', dtype=tf.int64)
        self.tp_plus_fn = self.add_weight(name="tp_plus_fn", initializer='zeros', dtype=tf.int64)
        self.tp_plus_fp = self.add_weight(name="tp_plus_fp", initializer='zeros', dtype=tf.int64)

    def update_state(self, y_true, y_pred):
        class_predictions = tf.math.argmax(y_pred, axis=-1)
        y_true = tf.where(y_true != self.label_to_ignore, y_true, self.num_classes)
        true_positive_weights_mask = tf.cast(class_predictions == y_true, dtype=tf.int64)
        self.tp_plus_fp += tf.math.bincount(class_predictions, minlength=self.num_classes, maxlength=self.num_classes)
        self.tp += tf.math.bincount(class_predictions, minlength=self.num_classes, maxlength=self.num_classes,
                                                weights=true_positive_weights_mask)
        self.tp_plus_fn += tf.math.bincount(y_true, minlength=self.num_classes, maxlength=self.num_classes)

    def result(self):
        precisions = tf.cast(self.tp, tf.float32) / tf.cast(tf.clip_by_value(self.tp_plus_fp, 1, 1e12), tf.float32)
        recalls = tf.cast(self.tp, tf.float32) / tf.cast(tf.clip_by_value(self.tp_plus_fn, 1, 1e12), tf.float32)
        f1_scores = (2 * precisions * recalls) / tf.clip_by_value(precisions + recalls, 1., float(1e12))
        return {f"{class_name}_f1": f1_scores[i] for i, class_name in self.label_id_to_name.items()}