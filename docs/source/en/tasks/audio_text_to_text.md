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

Audio-text-to-text models, are language models that take audio and text inputs and generate text outputs. These models combine audio understanding with language generation capabilities, enabling tasks like audio question answering, audio reasoning, and spoken dialogue understanding. Unlike traditional ASR models that simply transcribe speech, audio-text-to-text models can reason about audio content, follow instructions, and generate contextual responses.

This guide will show you how to:

1. Fine-tune [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) on the [Big Bench Audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio) dataset for audio reasoning tasks using LoRA.
2. Use your fine-tuned model for inference.

<Tip>

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/audio-text-to-text)

</Tip>

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers datasets soundfile peft accelerate
```

We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load Big Bench Audio dataset

Start by loading the [Big Bench Audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio) dataset from the 🤗 Datasets library. This dataset contains 1000 samples of audio reasoning tasks across four categories:

- **Formal Fallacies**: Logical deduction evaluation
- **Navigate**: Navigation step determination
- **Object Counting**: Item counting within collections
- **Web of Lies**: Boolean logic evaluation

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("ArtificialAnalysis/big_bench_audio", split="train")
```

Cast the audio column to 16kHz, which is required by Voxtral:

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

Shuffle and split the dataset into train and test sets:

```py
>>> dataset = dataset.shuffle(seed=42)
>>> train_samples = len(dataset) - 100  # 900 for training
>>> train_dataset = dataset.select(range(train_samples))
>>> eval_dataset = dataset.select(range(train_samples, len(dataset)))
```

Take a look at an example:

```py
>>> train_dataset[0]
{'audio': {'array': array([...], dtype=float32),
  'path': '...',
  'sampling_rate': 16000},
 'category': 'formal_fallacies',
 'official_answer': 'valid'}
```

The dataset contains:

- `audio`: the audio waveform containing the reasoning task
- `category`: the type of reasoning task
- `official_answer`: the expected answer

## Preprocess

Load the Voxtral processor to handle both audio and text inputs:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")

>>> # Ensure pad token is set
>>> if processor.tokenizer.pad_token is None:
...     processor.tokenizer.pad_token = processor.tokenizer.eos_token
```

Create a data collator that processes audio-text pairs into the format expected by Voxtral. The collator converts audio to base64 format and applies the chat template:

```py
>>> import torch
>>> import base64
>>> import io
>>> import soundfile as sf


>>> def audio_array_to_base64(audio_array, sampling_rate):
...     """Convert audio numpy array to base64 encoded WAV string."""
...     buffer = io.BytesIO()
...     sf.write(buffer, audio_array, sampling_rate, format='WAV')
...     buffer.seek(0)
...     audio_bytes = buffer.read()
...     return base64.b64encode(audio_bytes).decode('utf-8')


>>> class VoxtralAudioCollator:
...     """Data collator for Voxtral audio understanding training."""
...     
...     def __init__(self, processor):
...         self.processor = processor
...     
...     def __call__(self, features):
...         # Category-specific questions
...         category_questions = {
...             "formal_fallacies": "Listen to the logical argument and determine if the conclusion is valid. Answer with the final conclusion.",
...             "navigate": "Listen to the navigation instructions and determine the final direction you are facing.",
...             "object_counting": "Listen carefully and count the items mentioned. Answer with the count.",
...             "web_of_lies": "Listen to the statements and determine whether the final person is telling the truth or lying.",
...         }
...         
...         batch_conversations = []
...         answers = []
...         
...         for f in features:
...             audio_array = f["audio"]["array"]
...             sampling_rate = f["audio"]["sampling_rate"]
...             audio_base64 = audio_array_to_base64(audio_array, sampling_rate)
...             
...             category = f.get("category", "")
...             question = category_questions.get(category, "Listen to the audio and answer the question.")
...             answer = f.get("official_answer", "")
...             answers.append(answer)
...             
...             # Build conversation format for Voxtral
...             conversation = [
...                 {
...                     "role": "user",
...                     "content": [
...                         {"type": "audio", "base64": f"data:audio/wav;base64,{audio_base64}"},
...                         {"type": "text", "text": question},
...                     ],
...                 }
...             ]
...             batch_conversations.append(conversation)
...         
...         # Process conversations
...         inputs = self.processor.apply_chat_template(
...             batch_conversations,
...             tokenize=True,
...             return_tensors="pt",
...             padding=True,
...         )
...         
...         prompt_ids = inputs["input_ids"]
...         prompt_attn = inputs["attention_mask"]
...         B = prompt_ids.size(0)
...         tok = self.processor.tokenizer
...         
...         # Tokenize answers
...         answer_tok = tok(
...             answers,
...             add_special_tokens=False,
...             padding=False,
...             truncation=True,
...             max_length=512,
...             return_tensors=None,
...         )
...         answer_ids_list = answer_tok["input_ids"]
...         
...         # Build sequences: [PROMPT] + [ANSWER] + [EOS]
...         input_ids, attention_mask, labels = [], [], []
...         passthrough = {k: v for k, v in inputs.items() 
...                        if k not in ("input_ids", "attention_mask")}
...         
...         for i in range(B):
...             p_ids = prompt_ids[i].tolist()
...             p_att = prompt_attn[i].tolist()
...             a_ids = answer_ids_list[i]
...             
...             # Remove padding from prompt
...             non_pad_len = sum(p_att)
...             p_ids = p_ids[:non_pad_len]
...             p_att = p_att[:non_pad_len]
...             
...             # Concatenate: prompt + answer + eos
...             ids = p_ids + a_ids + [tok.eos_token_id]
...             attn = p_att + [1] * (len(a_ids) + 1)
...             
...             # Labels: mask prompt (-100), learn only answer
...             lab = [-100] * len(p_ids) + a_ids + [tok.eos_token_id]
...             
...             input_ids.append(ids)
...             attention_mask.append(attn)
...             labels.append(lab)
...         
...         # Pad to max length
...         pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
...         max_len = max(len(x) for x in input_ids)
...         
...         def pad_to(seq, fill, length):
...             return seq + [fill] * (length - len(seq))
...         
...         input_ids = [pad_to(x, pad_id, max_len) for x in input_ids]
...         attention_mask = [pad_to(x, 0, max_len) for x in attention_mask]
...         labels = [pad_to(x, -100, max_len) for x in labels]
...         
...         batch = {
...             "input_ids": torch.tensor(input_ids, dtype=torch.long),
...             "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
...             "labels": torch.tensor(labels, dtype=torch.long),
...         }
...         
...         for k, v in passthrough.items():
...             batch[k] = v
...         
...         return batch
```

Instantiate the data collator:

```py
>>> data_collator = VoxtralAudioCollator(processor)
```

## Train

<Tip>

If you aren't familiar with finetuning a model with the [`Trainer`], take a look at the basic tutorial [here](../training#train-with-pytorch-trainer)!

</Tip>

Load the Voxtral model. We use `bfloat16` precision and `device_map="auto"` for efficient memory usage:

```py
>>> from transformers import VoxtralForConditionalGeneration

>>> model = VoxtralForConditionalGeneration.from_pretrained(
...     "mistralai/Voxtral-Mini-3B-2507",
...     torch_dtype=torch.bfloat16,
...     device_map="auto",
... )
```

Freeze the audio encoder to preserve pretrained audio representations:

```py
>>> for param in model.audio_tower.parameters():
...     param.requires_grad = False
```

### Configure LoRA

[LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) (Low-Rank Adaptation) enables efficient fine-tuning by only training a small number of additional parameters. Configure LoRA to target the language model's attention layers and the multi-modal projector:

```py
>>> from peft import LoraConfig, get_peft_model

>>> lora_config = LoraConfig(
...     r=8,
...     lora_alpha=32,
...     lora_dropout=0.05,
...     bias="none",
...     target_modules=[
...         # Language model attention
...         "q_proj", "k_proj", "v_proj", "o_proj",
...         # Feed-forward layers
...         "gate_proj", "up_proj", "down_proj",
...     ],
...     task_type="CAUSAL_LM",
... )

>>> model = get_peft_model(model, lora_config)
>>> model.print_trainable_parameters()
```

<Tip>

Including the multi-modal projector in LoRA targets is important for audio understanding tasks, as it helps the model better align audio features with the language model's representations.

</Tip>

### Setup training

Define training hyperparameters in [`TrainingArguments`]:

```py
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(
...     output_dir="voxtral-audio-reasoning-lora",
...     per_device_train_batch_size=2,
...     per_device_eval_batch_size=2,
...     gradient_accumulation_steps=8,
...     learning_rate=2e-4,
...     num_train_epochs=3,
...     bf16=True,
...     logging_steps=10,
...     eval_steps=100,
...     save_steps=100,
...     eval_strategy="steps",
...     save_strategy="steps",
...     warmup_ratio=0.1,
...     weight_decay=0.01,
...     lr_scheduler_type="cosine",
...     remove_unused_columns=False,
...     dataloader_num_workers=4,
...     gradient_checkpointing=True,
...     optim="adamw_torch",
...     dataloader_pin_memory=True,
...     push_to_hub=True,
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

Save the model and LoRA adapter:

```py
>>> trainer.save_model()
>>> processor.save_pretrained("voxtral-audio-reasoning-lora")

>>> # Save LoRA adapter separately for easy loading
>>> model.save_pretrained("voxtral-audio-reasoning-lora/lora_adapter")
```

Once training is completed, share your model to the Hub:

```py
>>> trainer.push_to_hub()
```

## Inference

Now that you've fine-tuned the model, you can use it for audio reasoning tasks.

Load the fine-tuned model and processor:

```py
>>> from transformers import VoxtralForConditionalGeneration, AutoProcessor
>>> from peft import PeftModel
>>> import torch

>>> base_model = VoxtralForConditionalGeneration.from_pretrained(
...     "mistralai/Voxtral-Mini-3B-2507",
...     torch_dtype=torch.bfloat16,
...     device_map="auto",
... )
>>> model = PeftModel.from_pretrained(base_model, "your-username/voxtral-audio-reasoning-lora/lora_adapter")
>>> processor = AutoProcessor.from_pretrained("your-username/voxtral-audio-reasoning-lora")
```

Load an audio sample for inference:

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("ArtificialAnalysis/big_bench_audio", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sample = dataset[0]
```

Prepare the input with a conversation format:

```py
>>> audio_base64 = audio_array_to_base64(sample["audio"]["array"], sample["audio"]["sampling_rate"])

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "audio", "base64": f"data:audio/wav;base64,{audio_base64}"},
...             {"type": "text", "text": "Listen to the audio and answer the question."},
...         ],
...     }
... ]

>>> inputs = processor.apply_chat_template(
...     messages,
...     tokenize=True,
...     return_tensors="pt",
...     add_generation_prompt=True,
... ).to(model.device)
```

Generate a response:

```py
>>> with torch.no_grad():
...     output_ids = model.generate(**inputs, max_new_tokens=100)

>>> # Decode only the generated tokens
>>> input_len = inputs["input_ids"].shape[1]
>>> response = processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
>>> print(response)
```

## Pipeline

You can also use the [`Pipeline`] API for quick inference. Instantiate a pipeline for audio-text-to-text:

```py
>>> from transformers import pipeline

>>> pipe = pipeline(
...     "audio-text-to-text",
...     model="your-username/voxtral-audio-reasoning-lora",
... )

>>> result = pipe(
...     sample["audio"]["array"],
...     generate_kwargs={"max_new_tokens": 100},
... )
>>> print(result["generated_text"])
```

<Tip>

For more advanced use cases like multi-turn conversations with audio, you can structure your messages with alternating user and assistant roles, similar to [image-text-to-text](./image_text_to_text) models.

</Tip>

## Further Reading

- [Audio-text-to-text task page](https://huggingface.co/tasks/audio-text-to-text) covers model types, use cases, and datasets.
- [PEFT documentation](https://huggingface.co/docs/peft) for more LoRA configuration options and other adapter methods.
- [Voxtral model card](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) for model-specific details and capabilities.
