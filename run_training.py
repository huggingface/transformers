#!/usr/bin/env python3
import datasets

import soundfile as sf
from transformers import DataCollatorCTCWithPadding, Trainer, TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2Tokenizer


model = Wav2Vec2ForCTC.from_pretrained("./hf/wav2vec2-base")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("./hf/wav2vec2-base")

dummy_speech_data = datasets.load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
dummy_speech_data = dummy_speech_data.select(range(32))


def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch


dummy_speech_data = dummy_speech_data.map(map_to_array, remove_columns=["file"])


def prepare_dataset(batch):
    batch["input_values"] = tokenizer(batch["speech"]).input_values
    batch["labels"] = tokenizer.encode_labels(batch["text"]).labels
    return batch


data_collator = DataCollatorCTCWithPadding(tokenizer=tokenizer, padding=True)

train_dataset = dummy_speech_data.map(prepare_dataset, batch_size=16, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    weight_decay=0.01,
    learning_rate=1e-4,
    warmup_steps=3000,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
