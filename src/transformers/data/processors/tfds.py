# coding=utf-8
"""Data processors for Tensorflow datasets"""

import csv

import tensorflow_datasets.public_api as tfds
import tensorflow as tf


class SequenceClassification(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version(
        "1.0.0", "Sequence Classification dataset builder",
        experiments={tfds.core.Experiment.S3: False})

    def __init__(self, **config):
        self.name = config.pop("dataset_name", None)
        self.files = {}
        self.files["train"] = config.pop("train_file", None)
        self.files["dev"] = config.pop("dev_file", None)
        self.files["test"] = config.pop("test_file", None)
        self.is_column_id = config.pop("is_column_id", False)
        self.skip_first_row = config.pop("skip_first_row", True)
        self.delimiter = config.pop("delimiter", "\t")
        self.quotechar = config.pop("quotechar", "\"")
        self.labels = []
        super(SequenceClassification, self).__init__(**config)

    def _info(self):
        features = {
            "guid": tf.int32,
            "text_a": tfds.features.Text(),
            "text_b": tfds.features.Text(),
            "label": tfds.features.ClassLabel(num_classes=None),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description="Generic sequence classification dataset.",
            features=tfds.features.FeaturesDict(features),
        )

    def _split_generators(self, dl_manager):
        labels = set()

        for split in ["train", "dev", "test"]:
            if self.files[split] is None:
                continue

            lines, columns = read_csv(self.files[split], self.is_column_id, self.delimiter, self.quotechar, self.skip_first_row)
            column_label = columns[1]

            for line in lines:
                labels.add(line[column_label])

        self.info.features["label"].names = list(labels)

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"split": "train"},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={"split": "dev"},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"split": "test"},
            ),
        ]

    def _generate_examples(self, split):
        if not self.files[split]:
            return None, {}

        lines, columns = read_csv(self.files[split], self.is_column_id, self.delimiter, self.quotechar, self.skip_first_row)
        column_id = columns[0]
        column_label = columns[1]
        column_text_a = columns[2]
        column_text_b = columns[3]

        for (i, line) in enumerate(lines):
            if column_id == -1:
                id = i
            else:
                id = line[column_id]

            if column_text_b == -1:
                text_b = ""
            else:
                text_b = line[column_text_b]

            self.labels = list(set(self.labels).union(set([line[column_label]])))

            yield id, {
                "guid": id,
                "text_a": line[column_text_a],
                "text_b": text_b,
                "label": line[column_label]
            }


def read_csv(path, is_column_id, delimiter, quotechar, skip_first_row):
    with tf.io.gfile.GFile(path) as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)

        if skip_first_row:
            lines = list(reader)[1:]
        else:
            lines = list(reader)

    if len(lines[0]) == 3 and is_column_id:
        column_id = 0
        column_label = 1
        column_text_a = 2
        column_text_b = -1
    elif len(lines[0]) == 3 and not is_column_id:
        column_id = -1
        column_label = 0
        column_text_a = 1
        column_text_b = 2
    elif len(lines[0]) == 4 and is_column_id:
        column_id = 0
        column_label = 1
        column_text_a = 2
        column_text_b = 3
    elif len(lines[0]) == 2 and not is_column_id:
        column_id = -1
        column_label = 0
        column_text_a = 1
        column_text_b = -1
    else:
        raise csv.Error("The CSV file " + path + " is malformed")

    return lines, [column_id, column_label, column_text_a, column_text_b]
