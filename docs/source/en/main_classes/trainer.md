<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainer

The [`Trainer`] class provides an API for feature-complete training in PyTorch, and it supports distributed training on multiple GPUs/TPUs, mixed precision for [NVIDIA GPUs](https://nvidia.github.io/apex/), [AMD GPUs](https://rocm.docs.amd.com/en/latest/rocm.html), and [`torch.amp`](https://pytorch.org/docs/stable/amp.html) for PyTorch. [`Trainer`] goes hand-in-hand with the [`TrainingArguments`] class, which offers a wide range of options to customize how a model is trained. Together, these two classes provide a complete training API.

[`Seq2SeqTrainer`] and [`Seq2SeqTrainingArguments`] inherit from the [`Trainer`] and [`TrainingArguments`] classes and they're adapted for training models for sequence-to-sequence tasks such as summarization or translation.

<Tip warning={true}>

The [`Trainer`] class is optimized for 🤗 Transformers models and can have surprising behaviors
when used with other models. When using it with your own model, make sure:

- your model always return tuples or subclasses of [`~utils.ModelOutput`]
- your model can compute the loss if a `labels` argument is provided and that loss is returned as the first
  element of the tuple (if your model returns tuples)
- your model can accept multiple label arguments (use `label_names` in [`TrainingArguments`] to indicate their name to the [`Trainer`]) but none of them should be named `"label"`

</Tip>

## Trainer[[api-reference]]

[[autodoc]] Trainer
    - all

### Using a Custom Loss Function

By default, the `Trainer` uses the model's internal loss computation (if `labels` are provided). If you need to implement your own loss logic (e.g., for label smoothing, a special contrastive loss, etc.), the most robust method is to subclass `Trainer` and override the `compute_loss` method.

Here is an example of a trainer that implements a custom Cross-Entropy loss with label smoothing:

```python
from transformers import Trainer
from torch import nn

class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. Extract labels from the inputs dictionary
        labels = inputs.pop("labels")
        
        # 2. Get the standard model outputs (which will contain the logits)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 3. Define your custom loss function
        # In this example, we use CrossEntropyLoss with label smoothing
        loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 4. Compute the loss
        loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# You would then instantiate and use this trainer as usual
# trainer = CustomLossTrainer(...)
# trainer.train()
```
This approach cleanly separates your custom training logic from the model's architecture.

## Seq2SeqTrainer

[[autodoc]] Seq2SeqTrainer
    - evaluate
    - predict

## TrainingArguments

[[autodoc]] TrainingArguments
    - all

## Seq2SeqTrainingArguments

[[autodoc]] Seq2SeqTrainingArguments
    - all
