<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Audio-text-to-text

[[open-in-colab]]

Audio-text-to-text models accept both audio and text as inputs and generate text as output. They combine audio understanding with language generation, enabling tasks like audio question answering (e.g., "What is being said in this clip?"), audio reasoning (e.g., "What emotion does the speaker convey?"), and spoken dialogue understanding. Unlike traditional automatic speech recognition (ASR) models that only transcribe speech into text, audio-text-to-text models can reason about the audio content, follow complex instructions, and produce contextual responses based on what they hear.

The example below shows how to load a model and processor, pass an audio file with a text prompt, and generate a response. In this case, we ask the model to transcribe a speech recording.

```python
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the input speech."},
            {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/WhDJDIviAOg_120_10.mp3"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
## ["The transcription of the audio is 'summer follows spring the days grow longer and the nights are warm'."]
```

This guide will show you how to:

1. Fine-tune [Audio Flamingo 3](https://huggingface.co/nvidia/audio-flamingo-3-hf) on the [AudioCaps](https://huggingface.co/datasets/OpenSound/AudioCaps) dataset for audio captioning using LoRA.
2. Use your fine-tuned model for inference.

> [!TIP]
> To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/audio-text-to-text).


Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers datasets peft accelerate
```

We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login:

```py
>>> from huggingface_hub import notebook_login
>>> notebook_login()
```

## Load AudioCaps dataset

Start by loading the [AudioCaps](https://huggingface.co/datasets/OpenSound/AudioCaps) dataset from the 🤗 Datasets library in streaming mode. This dataset contains audio clips with descriptive captions, perfect for audio captioning tasks.

```py
>>> from datasets import load_dataset, Audio
>>> dataset = load_dataset("OpenSound/AudioCaps", split="train", streaming=True)
```

Cast the audio column to 16kHz, which is required by Audio Flamingo's Whisper feature extractor:

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

Split the dataset into train and test sets using `.take()` and `.skip()` for streaming datasets:

```py
>>> train_dataset = dataset.take(1000)
>>> eval_dataset = dataset.skip(1000).take(100)
```

Take a look at an example:

```py
>>> next(iter(train_dataset))
{'audio': {'array': array([...], dtype=float32),
  'path': '...',
  'sampling_rate': 16000},
 'caption': 'A man speaks followed by footsteps'}
```

The dataset contains:

- `audio`: the audio waveform
- `caption`: the descriptive text caption for the audio

## Preprocess

Load the Audio Flamingo processor to handle both audio and text inputs:

```py
>>> from transformers import AutoProcessor
>>> processor = AutoProcessor.from_pretrained("nvidia/audio-flamingo-3-hf")
```

Create a data collator that processes audio-text pairs into the format expected by Audio Flamingo. The collator uses the chat template format with direct audio arrays:

```py
>>> class AudioFlamingo3DataCollator:
...     """Data collator for Audio Flamingo 3 audio captioning training."""
...
...     def __init__(self, processor):
...         self.processor = processor
...
...     def __call__(self, features):
...         conversations = []
...
...         for feature in features:
...             # Build conversation format for Audio Flamingo
...             # Audio is passed directly as an array, no base64 encoding needed
...             sample = [
...                 {
...                     "role": "user",
...                     "content": [
...                         {"type": "text", "text": "Describe the audio."},
...                         {"type": "audio", "audio": feature["audio"]["array"]},
...                     ],
...                 },
...                 {
...                     "role": "assistant",
...                     "content": [{"type": "text", "text": feature["caption"]}],
...                 }
...             ]
...             conversations.append(sample)
...
...         # Apply chat template and format labels for training
...         return self.processor.apply_chat_template(
...             conversations,
...             tokenize=True,
...             add_generation_prompt=False,
...             return_dict=True,
...             output_labels=True,  # Automatically creates labels for training
...         )
```

Instantiate the data collator:

```py
>>> data_collator = AudioFlamingo3DataCollator(processor)
```

## Train

> [!TIP]
> If you aren't familiar with fine-tuning a model with the [`Trainer`], take a look at the basic tutorial [here](../training)!


Load the Audio Flamingo model. We use `bfloat16` precision and `device_map="auto"` for efficient memory usage:

```py
>>> from transformers import AudioFlamingo3ForConditionalGeneration
>>> import torch
>>> model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
...     "nvidia/audio-flamingo-3-hf",
...     torch_dtype=torch.bfloat16,
...     device_map="auto",
... )
```

### Configure LoRA

[LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) (Low-Rank Adaptation) enables efficient fine-tuning by only training a small number of additional parameters. Configure LoRA to target the language model's attention and feed-forward layers:

```py
>>> from peft import LoraConfig, get_peft_model
>>> lora_config = LoraConfig(
...     r=16,  # LoRA rank
...     lora_alpha=32,  # LoRA scaling factor
...     target_modules=[
...         # Language model attention
...         "q_proj",
...         "k_proj",
...         "v_proj",
...         "o_proj",
...         # Feed-forward layers
...         "gate_proj",
...         "up_proj",
...         "down_proj",
...     ],
...     lora_dropout=0.05,
...     bias="none",
...     task_type="CAUSAL_LM",
... )
>>> model = get_peft_model(model, lora_config)
>>> model.print_trainable_parameters()
```

> [!TIP]
> [LoRA](https://huggingface.co/docs/peft/main/conceptual_guides/lora) significantly reduces memory usage and training time by only updating a small number of adapter parameters instead of the full model. This configuration targets the language model's attention and feed-forward layers while keeping the audio encoder frozen, making it possible to fine-tune on a single GPU.


### Setup training

Define training hyperparameters in [`TrainingArguments`]. Note that we use `max_steps` instead of epochs since we're using a streaming dataset:

```py
>>> from transformers import TrainingArguments, Trainer
>>> training_args = TrainingArguments(
...     output_dir="audio-flamingo-3-hf-lora-finetuned",
...     per_device_train_batch_size=4,
...     per_device_eval_batch_size=4,
...     gradient_accumulation_steps=4,
...     learning_rate=1e-4,
...     max_steps=500,  # Use max_steps with streaming datasets
...     bf16=True,
...     logging_steps=10,
...     eval_steps=100,
...     save_steps=250,
...     save_total_limit=2,  # Keep only the latest 2 checkpoints
...     save_only_model=True,  # Skip saving optimizer state to save disk space
...     eval_strategy="steps",
...     save_strategy="steps",
...     remove_unused_columns=False,
...     dataloader_num_workers=0,  # Must be 0 for streaming datasets
...     gradient_checkpointing=True,
...     report_to="none",
... )
```

Pass the training arguments to [`Trainer`] along with the model, datasets, and data collator:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=train_dataset,
...     eval_dataset=eval_dataset,
...     data_collator=data_collator,
... )
>>> trainer.train()
```

Save the LoRA adapter and processor:

```py
>>> trainer.save_model()
>>> processor.save_pretrained("audio-flamingo-3-hf-lora-finetuned")
```

Once training is completed, share your model to the Hub:

```py
>>> trainer.push_to_hub()
```

## Inference

Now that you've fine-tuned the model, you can use it for audio captioning.

Load the fine-tuned model and processor:

```py
>>> from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
>>> from peft import PeftModel
>>> import torch
>>> base_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
...     "nvidia/audio-flamingo-3-hf",
...     torch_dtype=torch.bfloat16,
...     device_map="auto",
... )
>>> model = PeftModel.from_pretrained(base_model, "audio-flamingo-3-hf-lora-finetuned")
>>> processor = AutoProcessor.from_pretrained("audio-flamingo-3-hf-lora-finetuned")
```

Load an audio sample for inference:

```py
>>> from datasets import load_dataset, Audio
>>> dataset = load_dataset("OpenSound/AudioCaps", split="test", streaming=True)
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sample = next(iter(dataset))
```

Prepare the input with a conversation format:

```py
>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "text", "text": "Describe the audio."},
...             {"type": "audio", "audio": sample["audio"]["array"]},
...         ],
...     }
... ]
>>> inputs = processor.apply_chat_template(
...     messages,
...     tokenize=True,
...     add_generation_prompt=True,
...     return_dict=True,
... )
```

Generate a response:

```py
>>> with torch.no_grad():
...     output_ids = model.generate(**inputs, max_new_tokens=100)
>>> # Decode only the generated tokens
>>> input_len = inputs["input_ids"].shape[1]
>>> response = processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
>>> print(response)
## A sewing machine is running while people are talking
```

## Pipeline

You can also use the [`Pipeline`] API for quick inference. First, merge the LoRA adapter with the base model, then create a pipeline:

```py
>>> from transformers import pipeline
>>> # Merge LoRA adapter for pipeline use
>>> merged_model = model.merge_and_unload()
>>> pipe = pipeline(
...     "audio-text-to-text",
...     model=merged_model,
...     processor=processor,
... )
>>> result = pipe(
...     sample["audio"]["array"],
...     generate_kwargs={"max_new_tokens": 100},
... )
>>> print(result[0]["generated_text"])
```

> [!TIP]
> For more advanced use cases like multi-turn conversations with audio, you can structure your messages with alternating user and assistant roles, similar to [image-text-to-text](./image_text_to_text) models.


## Further Reading

- [Audio-text-to-text task page](https://huggingface.co/tasks/audio-text-to-text) covers model types, use cases, and datasets.
- [PEFT documentation](https://huggingface.co/docs/peft) for more LoRA configuration options and other adapter methods.
- [Audio Flamingo 3 model card](https://huggingface.co/nvidia/audio-flamingo-3-hf) for model-specific details and capabilities.
