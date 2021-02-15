#!/usr/bin/env python3
import datasets

import soundfile as sf
from transformers import DataCollatorCTCWithPadding, Trainer, TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import numpy as np

from jiwer import wer


model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base")

train_dataset = datasets.load_dataset("librispeech_asr", "clean", split="train.100")
val_dataset = datasets.load_dataset("librispeech_asr", "clean", split="validation")


def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch


train_dataset = train_dataset.map(map_to_array, remove_columns=["file"])
val_dataset = val_dataset.map(map_to_array, remove_columns=["file"])


def prepare_dataset(batch):
    batch["input_values"] = tokenizer(batch["speech"]).input_values
    batch["labels"] = tokenizer.encode_labels(batch["text"]).labels
    return batch


data_collator = DataCollatorCTCWithPadding(tokenizer=tokenizer, padding=True)

train_dataset = train_dataset.map(prepare_dataset, batch_size=12, batched=True, num_proc=8)
val_dataset = val_dataset.map(prepare_dataset, batch_size=16, batched=True, num_proc=8)


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = 0

    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(pred.label_ids)

    wer_score = wer(label_str, pred_str)

    return {"wer": wer_score}


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    gradient_accumulation_steps=4,
    save_total_limit=3,
    save_steps=200,
    eval_steps=50,
    logging_steps=10,
    learning_rate=1e-4,
    warmup_steps=2000,
    logging_dir="./logs",
    fp16=True,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
