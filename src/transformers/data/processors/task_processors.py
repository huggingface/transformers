# coding=utf-8
"""Data processors per task"""

import csv
import logging
from abc import ABC, abstractmethod

from .tfds import SequenceClassification  # noqa: F401
import tensorflow_datasets as tfds

from ...file_utils import is_tf_available, is_torch_available
from .utils import InputExample, InputFeatures


logger = logging.getLogger(__name__)


class CSVData(ABC):
    def __init__(self, **config):
        self.skip_first_row = config.pop("skip_first_row", True)
        self.delimiter = config.pop("delimiter", "\t")
        self.quotechar = config.pop("quotechar", "\"")
        self.is_column_id = config.pip("is_column_id", False)

    def read_csv(self, input_file):
        if input_file is None:
            return []

        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=self.delimiter, quotechar=self.quotechar)

            if self.skip_first_row:
                lines = list(reader)[1:]
            else:
                lines = list(reader)

            if len(lines[0]) == 3 and self.is_column_id:
                column_id = 0
                column_label = 1
                column_text_a = 2
                column_text_b = -1
            elif len(lines[0]) == 3 and not self.is_column_id:
                column_id = -1
                column_label = 0
                column_text_a = 1
                column_text_b = 2
            elif len(lines[0]) == 4 and self.is_column_id:
                column_id = 0
                column_label = 1
                column_text_a = 2
                column_text_b = 3
            elif len(lines[0]) == 2 and not self.is_column_id:
                column_id = -1
                column_label = 0
                column_text_a = 1
                column_text_b = -1
            else:
                raise csv.Error("The CSV file " + input_file + " is malformed")

        return lines, [column_id, column_label, column_text_a, column_text_b]


class TFDSData(ABC):
    def __init__(self, **config):
        self.guid = config.pop("guid", "guid")
        self.text_a = config.pop("text_a", "text_a")
        self.text_b = config.pop("text_b", "text_b")
        self.label = config.pop("label", "label")

    def get_example_from_tensor_dict(self, tensor_dict):
        """Get an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        return InputExample(
            tensor_dict[self.guid].numpy(),
            tensor_dict[self.text_a].numpy().decode("utf-8"),
            tensor_dict[self.text_b].numpy().decode("utf-8"),
            str(tensor_dict[self.label].numpy()),
        )


class DataProcessor(ABC):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, **config):
        self.labels = []
        self.examples = {}
        self.examples["train"] = []
        self.examples["validation"] = []
        self.examples["test"] = []
        self.files = {}
        self.files["train"] = config.pop("train_file", None)
        self.files["validation"] = config.pop("dev_file", None)
        self.files["test"] = config.pop("test_file", None)

        self._create_examples()

        # assert len(config) == 0, "unrecognized params passed: %s" % ",".join(config.keys())

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.labels

    def num_examples(self, mode):
        return len(self.examples[mode])

    @abstractmethod
    def _create_examples(self):
        pass

    def convert_examples_to_features(self, mode, tokenizer, max_len, return_dataset="tf"):
        if max_len is None:
            max_len = tokenizer.max_len

        label_map = {label: i for i, label in enumerate(self.labels)}
        features = []

        for (ex_index, example) in enumerate(self.examples[mode]):
            if ex_index % 10000 == 0:
                logger.info("Tokenizing example %d", ex_index)

            feature = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_len, pad_to_max_length=True)
            label = label_map[example.label]

            assert len(feature["input_ids"]) == max_len
            assert len(feature["attention_mask"]) == max_len
            assert len(feature["token_type_ids"]) == max_len

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in feature["input_ids"]]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in feature["attention_mask"]]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in feature["token_type_ids"]]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            features.append(InputFeatures(input_ids=feature["input_ids"],
                                          attention_mask=feature["attention_mask"],
                                          token_type_ids=feature["token_type_ids"],
                                          label=label))

        if len(features) == 0:
            return None

        if return_dataset == "tf":
            if not is_tf_available():
                raise RuntimeError("return_dataset set to 'tf' but TensorFlow 2.0 can't be imported")

            import tensorflow as tf

            def gen():
                for ex in features:
                    yield ({"input_ids": ex.input_ids, "attention_mask": ex.attention_mask, "token_type_ids": ex.token_type_ids}, ex.label)

            dataset = tf.data.Dataset.from_generator(
                gen,
                ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
                ({"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None]), "token_type_ids": tf.TensorShape([None])}, tf.TensorShape([])),
            )

            return dataset
        elif return_dataset == "pt":
            if not is_torch_available():
                raise RuntimeError("return_dataset set to 'pt' but PyTorch can't be imported")

            import torch
            from torch.utils.data import TensorDataset

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)

            return dataset
        else:
            raise ValueError("return_tensors should be one of 'tf' or 'pt'")


class DataProcessorForSequenceClassificationWithTFDS(DataProcessor, TFDSData):
    def __init__(self, **config):
        features = config.pop("features", {})
        TFDSData.__init__(self, **features)
        DataProcessor.__init__(self, **config)
        self.dataset_name = config.pop("dataset_name", None)

        if self.dataset_name in tfds.list_builders():
            self.ds, info = tfds.load(self.dataset_name, with_info=True)
        else:
            self.ds, info = tfds.load("sequence_classification",
                                      builder_kwargs={
                                          "train_file": self.train_file,
                                          "dev_file": self.dev_file,
                                          "test_file": self.test_file,
                                          "dataset_name": self.dataset_name})

        self.labels = info.features[features["label"]].names

    def _create_examples(self):
        for mode in ["train", "validation", "test"]:
            tf_dataset = self.ds[mode]

            for entry in tf_dataset:
                example = self.get_example_from_tensor_dict(entry)

                self.examples[mode].append(example)


class DataProcessorForSequenceClassificationWithCSV(DataProcessor, CSVData):
    """ Generic processor for sentence classification data set."""
    def __init__(self, **config):
        CSVData.__init__(self, **config)
        DataProcessor.__init__(self, **config)

    def _create_examples(self):
        for mode in ["train", "validation", "test"]:
            lines, columns = self.read_csv(self.files[mode])
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

                self.examples[mode].append(InputExample(guid=id, text_a=line[column_text_a], text_b=text_b, label=line[column_label]))
