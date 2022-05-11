from dataclasses import dataclass
from typing import Optional, Union

import torch

from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorForKeyValueExtraction(DataCollatorMixin):
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
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        pixel_values = None
        if "pixel_values" in features[0]:
            pixel_values = torch.stack([torch.tensor(d.pop("pixel_values")) for d in features])
            IMAGE_LEN = int(pixel_values.shape[-1] / 16) * int(pixel_values.shape[-1] / 16) + 1

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=512,
            pad_to_multiple_of=None,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
            batch = {
                k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == "attention_mask" else v
                for k, v in batch.items()
            }

        if labels is None:
            return batch

        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        padding_idx = self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]

        # pad on the right by default
        batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        if has_bbox_input:
            batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
        if has_position_input:
            batch["position_ids"] = [
                position_id + [padding_idx] * (sequence_length - len(position_id))
                for position_id in batch["position_ids"]
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}

        if pixel_values is not None:
            visual_labels = torch.ones((len(batch["input_ids"]), IMAGE_LEN), dtype=torch.long) * -100
            batch["labels"] = torch.cat([batch["labels"], visual_labels], dim=1)

        return batch
