import os

import torch
from clearml import Task
from datasets import load_dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# Set environment variables
os.environ["CLEARML_PROJECT"] = "Test Project"
os.environ["CLEARML_TASK"] = "Test Task"
os.environ["CLEARML_LOG_MODEL"] = "TRUE"

# Initialize ClearML task
task = Task.init(project_name="Test Project", task_name="Test Task", reuse_last_task_id=False)

# Load model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("glue", "mrpc")
dataset = dataset.map(
    lambda x: tokenizer(x["sentence1"], x["sentence2"], truncation=True),
    batched=True,
)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./test_output",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=100,
    report_to=["clearml"],
    logging_steps=10,
    save_total_limit=2,  # Limit number of saved checkpoints
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DataCollatorWithPadding(tokenizer),
)

# First training
trainer.train()

# Get task ID
task_id = task.id

# Verify checkpoint exists
checkpoint_path = "./test_output/checkpoint-459"  # Use last checkpoint
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

# Load model and training state from checkpoint
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
training_args = TrainingArguments(
    output_dir="./test_output",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=100,
    report_to=["clearml"],
    logging_steps=10,
    save_total_limit=2,
    resume_from_checkpoint=checkpoint_path,  # Resume directly from checkpoint
)

# Create new trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DataCollatorWithPadding(tokenizer),
)

# Continue training
trainer.train()

# Verify model parameters changed
initial_params = {name: param.data.clone() for name, param in model.named_parameters()}
trainer.train()
for name, param in model.named_parameters():
    if "weight" in name:
        diff = torch.abs(param.data - initial_params[name]).mean().item()
