<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Fine-tuning

Fine-tuning continues training a large pretrained model on a smaller dataset specific to a task or domain. For example, fine-tuning on a dataset of coding examples helps the model get better at coding. Fine-tuning is identical to pretraining except you don't start with random weights. It also requires far less compute, data, and time.

The tutorial below walks through fine-tuning a large language model with [`Trainer`].

Log in to your Hugging Face account with your user token to push your fine-tuned model to the Hub.

```py
from huggingface_hub import login

login()
```

## Tokenization

Load a dataset and [tokenize](./fast_tokenizers) the text column the model trains on (`horoscope` in the dataset below).

<iframe
  src="https://huggingface.co/datasets/karthiksagarn/astro_horoscope/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

The tokenizer creates the model inputs, `input_ids` and `attention_mask`. The model's forward method only accepts `input_ids` and `attention_mask`, so set `remove_columns` to drop columns like `horoscope` after tokenization.

- Set `truncation=True` and a `max_length` to truncate longer sequences to a specified maximum length.
- Use the [`~datasets.train_test_split`] method to create a test split for evaluating the model.

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)
```

A data collator assembles dataset samples into batches for the model to process. [`DataCollatorForLanguageModeling`] *dynamically* pads each batch to the longest sequence in that batch rather than padding every sequence in the dataset to the same length. This saves compute and memory by avoiding computing unnecessary padding tokens.

- Set `mlm=False` to avoid randomly masking tokens.

```py
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
```

## Loading a model

Load a pretrained checkpoint to fine-tune (see the [Loading models](./models) guide for more details about loading models).

- Set `dtype="auto"` to load the weights in their saved dtype. Without it, PyTorch loads weights in `torch.float32`, which doubles memory usage if the weights are originally `torch.bfloat16`.

```py
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
```

## Training configuration

[`TrainingArguments`] provides all the options for customizing a training run. Only the most common arguments are covered here. Everything else has reasonable defaults or is only relevant to specific scenarios like distributed training. See the [`TrainingArguments`] API docs for a complete list of arguments.

<hfoptions id="training-args">
<hfoption id="training duration">

- `num_train_epochs` and `per_device_train_batch_size` control training duration and batch size. `learning_rate` sets the initial learning rate for the optimizer.

</hfoption>
<hfoption id="training optimizations">

- Set `bf16=True` for fast mixed precision training if your hardware supports it (Ampere+ GPUs). Otherwise, fall back to `fp16=True` on older hardware.
- `gradient_accumulation_steps` simulates a larger effective batch size by accumulating gradients over multiple forward passes before updating weights.
- `gradient_checkpointing` trades compute for memory by recomputing intermediate activations during the backward pass instead of storing them.

</hfoption>
<hfoption id="evaluation and checkpointing">

- `eval_strategy` and `save_strategy` determine when to evaluate a model during training and when to save a checkpoint.
- `load_best_model_at_end` loads the best checkpoint when training finishes. It requires `eval_strategy` to be set.

</hfoption>
<hfoption id="logging">

- `logging_steps` controls how frequently to update and return loss during training.

</hfoption>
</hfoptions>

```py
TrainingArguments(
    output_dir="qwen3-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

## Training

Create a [`Trainer`] instance with all the necessary components, then call [`~Trainer.train`] to begin.

```py
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.push_to_hub()
```

[`~Trainer.push_to_hub`] uploads the fine-tuned weights, generation config, tokenizer, and model config to the Hub.

## Next steps

- Read the [Subclassing Trainer methods](./trainer_customize) guide to learn how to subclass [`Trainer`] methods to support new and custom functionalities.
- Read the [Callbacks](./trainer_callbacks) guide to learn how to hook into training events for logging, early stopping, and other custom behavior.
- Read the [Data collators](./data_collators) guide to learn how to customize how samples are assembled into batches.
- Browse [transformers/examples/pytorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), [notebooks](./notebooks), or the **Resources > Task Recipes** section for additional training examples on different text, audio, vision, and multimodal tasks.
