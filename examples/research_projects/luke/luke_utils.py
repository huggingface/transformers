import unicodedata
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from transformers.data.data_collator import DataCollatorMixin
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def padding_tensor(sequences, padding_value, padding_side, sequence_length):
    if isinstance(padding_value, tuple):
        out_tensor = np.full((len(sequences), sequence_length, 2), padding_value)
    else:
        out_tensor = np.full((len(sequences), sequence_length), padding_value)

    for i, tensor in enumerate(sequences):
        if padding_side == "right":
            if isinstance(padding_value, tuple):
                out_tensor[i, : len(tensor[:sequence_length]), :2] = tensor[:sequence_length]
            else:
                out_tensor[i, : len(tensor[:sequence_length])] = tensor[:sequence_length]
        else:
            if isinstance(padding_value, tuple):
                out_tensor[i, len(tensor[:sequence_length]) - 1 :, :2] = tensor[:sequence_length]
            else:
                out_tensor[i, len(tensor[:sequence_length]) - 1 :] = tensor[:sequence_length]

    return out_tensor.tolist()


def is_punctuation(char):
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


@dataclass
class DataCollatorForLukeTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["entity_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        ner_tags = [feature["ner_tags"] for feature in features]
        batch["ner_tags"] = padding_tensor(ner_tags, -1, padding_side, sequence_length)
        original_entity_spans = [feature["original_entity_spans"] for feature in features]
        batch["original_entity_spans"] = padding_tensor(original_entity_spans, (-1, -1), padding_side, sequence_length)
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        return batch
