from datasets import load_dataset

dataset = load_dataset("yelp_review_full")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("flaubert/flaubert_small_cased", num_labels=5)

from transformers import TrainingArguments


import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
                                    output_dir="test_trainer",
                                    evaluation_strategy="epoch",
                                    report_to="clearml",
                                    num_train_epochs=2,
                                    per_device_train_batch_size=5,
                                    logging_strategy="epoch",
                                    # logging_steps=1,
                                    save_strategy='epoch'
                                )

trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=small_train_dataset,
                    eval_dataset=small_eval_dataset,
                    compute_metrics=compute_metrics
                )


trainer.train()