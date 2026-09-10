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

# Transformers Quickstart Guide (বাংলা)

[[open-in-colab]]

Transformers-কে এমনভাবে ডিজাইন করা হয়েছে যাতে এটি দ্রুত আর সহজে ব্যবহার করা যায়, আর সবাই খুব সহজেই ট্রান্সফর্মার মডেল নিয়ে কাজ বা শেখা শুরু করে দিতে পারে।

ইউজারদের বোঝার সুবিধার্থে এখানে অ্যাবস্ট্রাকশন বা জটিলতা একদম কমিয়ে আনা হয়েছে—একটি মডেল ইন্সট্যানশিয়েট করার জন্য রয়েছে মাত্র তিনটি ক্লাস, আর ইনফারেন্স ও ট্রেনিংয়ের জন্য রয়েছে দুটি সহজ API।

এই কুইকস্টার্ট গাইডে Transformers-এর মূল ফিচারগুলোর সাথে পরিচিত হওয়ার পাশাপাশি আমরা শিখবো কীভাবে:

- একটি প্রিট্রেইন্ড মডেল লোড করা যায়
- `Pipeline` ব্যবহার করে ইনফারেন্স রান করা যায়
- `Trainer` দিয়ে একটি মডেলকে ফাইন-টিউন করা যায়

---

# সেটআপ (Set up)

শুরু করার জন্য প্রথমে একটি Hugging Face অ্যাকাউন্ট তৈরি করে নেওয়ার পরামর্শ দেব।

একটি অ্যাকাউন্ট থাকলে তুমি Hugging Face Hub-এ নিজের মডেল, ডাটাসেট আর Spaces হোস্ট করার পাশাপাশি সেগুলোর ভার্সন কন্ট্রোলও করতে পারবে।

এটি মূলত নতুন কিছু শেখা আর নতুন প্রজেক্ট বিল্ড করার একটি collaborative platform।

---

# Hugging Face Login

## Notebook Login

```python
from huggingface_hub import notebook_login

notebook_login()
```

---

## CLI Login

```bash
hf auth login
```

লগ ইনের প্রম্পট এলে নিজের User Access Token পেস্ট করে দাও।

---

# PyTorch Install

```bash
pip install torch
```

---

# Transformers Ecosystem Install

```bash
pip install -U transformers datasets evaluate accelerate timm
```

---

# Pretrained Models

প্রতিটি প্রিট্রেইন্ড মডেল তিনটি বেস ক্লাসের ওপর ভিত্তি করে কাজ করে।

| Class | Description |
|---|---|
| `PreTrainedConfig` | মডেলের configuration define করে |
| `PreTrainedModel` | মডেলের architecture define করে |
| `Preprocessor` | Input-কে tensor-এ convert করে |

---

# AutoClass API

মডেল এবং প্রিপ্রসেসর লোড করার জন্য AutoClass API ব্যবহার করা সবচেয়ে ভালো practice।

এটি automatically সঠিক architecture select করে নেয়।

---

# Model Load করা

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf"
)
```

---

# Important Parameters

## `device_map="auto"`

এটি automatically fastest available device (যেমন GPU)-এ model load করে।

---

## `dtype="auto"`

মডেল যেই datatype-এ save করা আছে সেই datatype-এই load হয়।

এটি memory efficient।

---

# Tokenization

```python
model_inputs = tokenizer(
    ["The secret to baking a good cake is "],
    return_tensors="pt"
).to(model.device)
```

---

# Text Generation

```python
generated_ids = model.generate(
    **model_inputs,
    max_length=30
)

output = tokenizer.batch_decode(generated_ids)[0]

print(output)
```

Expected Output:

```text
<s> The secret to baking a good cake is 100% in the preparation...
```

---

# Pipeline

Pretrained model দিয়ে inference run করার সবচেয়ে সহজ উপায় হলো `pipeline()` API।

এটি support করে:

- Text generation
- Image segmentation
- Speech recognition
- Question answering
- Summarization
- Translation

---

# Text Generation Pipeline

```python
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    device=device
)
```

---

# Generate Text

```python
result = pipe(
    "The secret to baking a good cake is ",
    max_length=50
)

print(result)
```

Expected Output:

```python
[
  {
    'generated_text': 'The secret to baking a good cake is 100% in the batter...'
  }
]
```

---

# Image Segmentation Pipeline

```python
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipe = pipeline(
    "image-segmentation",
    model="facebook/detr-resnet-50-panoptic",
    device=device
)
```

---

# Image Segmentation Example

```python
segments = pipe(
    "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
)

print(segments[0]["label"])
print(segments[1]["label"])
```

Expected Output:

```text
bird
bird
```

---

# Speech Recognition Pipeline

```python
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=device
)
```

---

# Speech Recognition Example

```python
result = pipe(
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
)

print(result)
```

Expected Output:

```python
{
  'text': 'He hoped there would be stew for dinner...'
}
```

---

# Trainer

`Trainer` হলো PyTorch model-এর complete training and evaluation loop।

এটি training boilerplate code অনেক কমিয়ে দেয়।

---

# Required Components

Trainer ব্যবহার করতে যা লাগবে:

- Model
- Dataset
- Tokenizer
- Data collator
- TrainingArguments

---

# Load Model and Dataset

```python
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased"
)

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased"
)

dataset = load_dataset("rotten_tomatoes")
```

---

# Tokenize Dataset

```python
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])

dataset = dataset.map(
    tokenize_dataset,
    batched=True
)
```

---

# Data Collator

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)
```

---

# Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=True,
)
```

---

# Trainer Setup

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)
```

---

# Start Training

```python
trainer.train()
```

---

# Push Model to Hub

```python
trainer.push_to_hub()
```

---

# Congratulations

অভিনন্দন! 🎉

Transformers দিয়ে তোমার প্রথম মডেল ট্রেইন করার কাজ সফলভাবে শেষ হয়েছে।

---

# Next Steps

## Base Classes

Configuration, model এবং processor classes নিয়ে আরও বিস্তারিত জানো।

---

## Inference

Pipeline, agents এবং optimized inference শেখো।

---

## Training

Distributed training এবং hardware optimization শেখো।

---

# Quantization

Quantization ব্যবহার করে:

- Memory usage কমানো যায়
- Storage কম লাগে
- Inference speed বাড়ে

---

# Resources

নির্দিষ্ট task-এর জন্য end-to-end recipes দেখতে:

- Hugging Face documentation
- Task recipes
- Official tutorials

---

# Useful Libraries

| Library | Purpose |
|---|---|
| transformers | NLP models |
| datasets | Dataset loading |
| evaluate | Evaluation |
| accelerate | Multi-GPU training |
| timm | Vision models |

---

# Best Practices

- GPU ব্যবহার করো
- Mixed precision training ব্যবহার করো
- Proper batching করো
- Dataset preprocessing optimize করো
- Smaller model দিয়ে শুরু করো

---

# Useful Links

- https://huggingface.co/
- https://huggingface.co/docs/transformers
- https://github.com/huggingface/transformers
- https://pytorch.org/

---

# Summary

এই guide-এ আমরা শিখলাম:

- Hugging Face setup
- Login system
- Pretrained models
- AutoClass API
- Tokenization
- Text generation
- Pipelines
- Speech recognition
- Image segmentation
- Trainer API
- Fine-tuning
- Push to Hub

এখন তুমি Transformers ব্যবহার করে NLP, Computer Vision, Speech AI এবং LLM-based applications build করতে পারবে।
