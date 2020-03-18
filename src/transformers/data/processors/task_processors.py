# coding=utf-8
"""Data processors per task"""

import copy
import csv
import json
import logging
from abc import ABC, abstractmethod

from ...file_utils import is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class CSVData(ABC):
    def __init__(self, **config):
        self.skip_first_row = config.pop("skip_first_row", False)
        self.delimiter = config.pop("delimiter", "\t")
        self.quotechar = config.pop("quotechar", None)

    def read_csv(self, input_file):
        if input_file is None:
            return []

        with open(input_file, "r", encoding="utf-8-sig") as f:
            if self.skip_first_row:
                return list(csv.reader(f, delimiter=self.delimiter, quotechar=self.quotechar))[1:]

            return list(csv.reader(f, delimiter=self.delimiter, quotechar=self.quotechar))

    @abstractmethod
    def _create_examples(self, lines, mode):
        pass


class TFDSData(ABC):
    def __init__(self, **config):
        self.guid = config.pop("guid", None)
        self.text_a = config.pop("text_a", None)
        self.text_b = config.pop("text_b", None)
        self.label = config.pop("label", None)

    def get_example_from_tensor_dict(self, tensor_dict, guid):
        """Get an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
            guid: ID of the given example
        """
        guid = guid if self.guid is None else tensor_dict[self.guid].numpy()
        text_b = self.text_b if self.text_b is None else tensor_dict[self.text_b].numpy().decode("utf-8")

        return InputExample(
            guid,
            tensor_dict[self.text_a].numpy().decode("utf-8"),
            text_b,
            str(tensor_dict[self.label].numpy()),
        )

    @abstractmethod
    def _create_examples(self, mode):
        pass


class DataProcessor(ABC):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, labels=None, train_examples=None, dev_examples=None, test_examples=None, **config):
        self.labels = [] if labels is None else labels
        self.examples = {}
        self.examples["train"] = [] if train_examples is None else train_examples
        self.examples["dev"] = [] if dev_examples is None else dev_examples
        self.examples["test"] = [] if test_examples is None else test_examples
        self.train_file = config.pop("train_file", None)
        self.dev_file = config.pop("dev_file", None)
        self.test_file = config.pop("test_file", None)

        self.create_train_examples()
        self.create_dev_examples()
        self.create_test_examples()

        # assert len(config) == 0, "unrecognized params passed: %s" % ",".join(config.keys())

    @abstractmethod
    def create_train_examples(self):
        """Create a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def create_dev_examples(self):
        """Create a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def create_test_examples(self):
        """Create a collection of `InputExample`s for the test set."""
        pass

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.labels

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
    def __init__(self, labels=None, train_examples=None, dev_examples=None, test_examples=None, **config):
        TFDSData.__init__(self, **config)
        DataProcessor.__init__(self, labels, train_examples, dev_examples, test_examples, **config)

    def create_train_examples(self):
        self._create_examples("train")

    def create_dev_examples(self):
        self._create_examples("dev")

    def create_test_examples(self):
        self._create_examples("test")

    def _create_examples(self, mode):
        td_dataset = self.examples[mode]
        examples = []
        added_labels = set()

        for (ex_index, entry) in enumerate(td_dataset):
            example = self.get_example_from_tensor_dict(entry, str(entry))

            added_labels.add(example.label)
            examples.append(example)

        self.labels = list(added_labels)
        self.examples[mode] = examples


class DataProcessorForSequenceClassificationWithCSV(DataProcessor, CSVData):
    """ Generic processor for sentence classification data set."""
    def __init__(self, labels=None, train_examples=None, dev_examples=None, test_examples=None, **config):
        self.column_label = config.pop("column_label", 0)
        self.column_text = config.pop("column_text", 1)
        self.column_id = config.pop("column_id", None)
        CSVData.__init__(self, **config)
        DataProcessor.__init__(self, labels, train_examples, dev_examples, test_examples, **config)

    def create_train_examples(self):
        lines = self.read_csv(self.train_file)

        self._create_examples(lines, "train")

    def create_dev_examples(self):
        lines = self.read_csv(self.dev_file)

        self._create_examples(lines, "dev")

    def create_test_examples(self):
        lines = self.read_csv(self.test_file)

        self._create_examples(lines, "test")

    def _create_examples(self, lines, mode):
        texts = []
        labels = []
        ids = []

        for (i, line) in enumerate(lines):
            texts.append(line[self.column_text])
            labels.append(line[self.column_label])

            if self.column_id is not None:
                ids.append(line[self.column_id])
            else:
                guid = "%s" % i
                ids.append(guid)

        assert len(texts) == len(labels)
        assert len(texts) == len(ids)

        added_labels = set()

        for (text, label, guid) in zip(texts, labels, ids):
            added_labels.add(label)
            self.examples[mode].append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

        self.labels = list(set(self.labels).union(added_labels))
