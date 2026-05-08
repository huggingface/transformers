<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Data collators

A data collator assembles individual dataset samples into a batch for the model. It can also dynamically pad samples to the longest sequence in *each batch*, which is more efficient than padding to a global maximum length.

```md
Dataset[0] → {"input_ids": [101, 2003], "labels": 1}
Dataset[1] → {"input_ids": [101, 2003, 1996], "labels": 0}
Dataset[2] → {"input_ids": [101, 7592], "labels": 1}
         ↓  collator
{
  "input_ids": tensor([[101, 2003,    0],   # padded to longest
                        [101, 2003, 1996],
                        [101, 7592,    0]]),
  "labels":    tensor([1, 0, 1])
}
```

Transformers provides data collators for various tasks (see all available [data collators](./main_classes/data_collator)). Create a custom data collator with:

- [DataCollatorWithPadding](#datacollatorwithpadding) when you need standard tokenizer-based padding plus extra fields.
- [DataCollatorMixin](#datacollatormixin) when you need custom padding logic, multiple paired inputs per sample, or a batch structure the tokenizer can't produce on its own.

## DataCollatorWithPadding

For simple use cases like adding an extra field, subclass [`DataCollatorWithPadding`] and extend its `__call__` method. The example below adds a `"score"` field.

1. Remove the custom field first because [`~PreTrainedTokenizerBase.pad`] doesn't recognize it.
2. Call the parent class to handle `input_ids` and `attention_mask`.
3. Add the `"score"` field back to the batch.

```py
import torch
from dataclasses import dataclass
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

@dataclass
class DataCollatorWithScore(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        scores = [f.pop("score") for f in features]

        batch = super().__call__(features)
        batch["score"] = torch.tensor(scores, dtype=torch.float)

        return batch
```

Pass the custom data collator to [`Trainer`] like any other data collator.

```py
trainer = Trainer(
    ...,
    data_collator=DataCollatorWithScore(tokenizer=tokenizer),
)
```

## DataCollatorMixin

Subclass [`DataCollatorMixin`] for full control over batch assembly and implement your own `__call__` method. Build custom padding logic, handle multiple input types, or create entirely new batch structures. The [DataCollatorForPreference](https://github.com/huggingface/trl/blob/cfbdd3bea4448cde878c0da0de49551f553c61fe/trl/trainer/reward_trainer.py#L126) example below uses [`DataCollatorMixin`] because each training sample has a chosen and rejected response, and the model needs to see both.

1. Separate `chosen_ids` and `rejected_ids` because [`~trl.trainer.utils.pad`] expects flat lists.
2. Concatenate the input pair into a single list.
3. Generate `attention_mask` with [torch.ones_like](https://docs.pytorch.org/docs/stable/generated/torch.ones_like.html) instead of the tokenizer because the collator works with raw token ID lists.
4. Pad `input_ids` and `attention_mask`.

```py
import torch
from transformers import DataCollatorMixin
from trl.trainer.utils import pad

class DataCollatorForPreference(DataCollatorMixin):
    pad_token_id: int
    pad_to_multiple_of: int | None = None

    def __call__(self, examples: list[dict]) -> dict:
        chosen_input_ids   = [torch.tensor(ex["chosen_ids"])   for ex in examples]
        rejected_input_ids = [torch.tensor(ex["rejected_ids"]) for ex in examples]

        input_ids      = chosen_input_ids + rejected_input_ids
        attention_mask = [torch.ones_like(ids) for ids in input_ids]

        output = {
            "input_ids": pad(
                input_ids,
                padding_value=self.pad_token_id,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ),
            "attention_mask": pad(
                attention_mask,
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ),
        }

        ...

        return output
```

## Next steps

- See all available [data collators](./main_classes/data_collator) for common tasks like token classification.
