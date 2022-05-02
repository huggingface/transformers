import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)
from transformers.file_utils import PaddingStrategy

@dataclass
class DataCollatorForKeyValueExtraction(DataCollatorMixin):
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
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v
                      for k, v in batch.items()}
            #visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            #batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)

        if labels is None:
            return batch

        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        padding_idx=self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        
        # pad on the right by default
        batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        if has_bbox_input:
            batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
        if has_position_input:
            batch["position_ids"] = [position_id + [padding_idx] * (sequence_length - len(position_id))
                                      for position_id in batch["position_ids"]]

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}

        if pixel_values is not None:
            visual_labels = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long) * -100
            batch["labels"] = torch.cat([batch['labels'], visual_labels], dim=1)

        return batch