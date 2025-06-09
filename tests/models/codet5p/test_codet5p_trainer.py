from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Use the correct model class with trust_remote_code
model_name = "Salesforce/codet5p-2b"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load a small dataset
dataset = load_dataset("glue", "mrpc", split="train[:10]")

# Preprocess the dataset
def preprocess(example):
    input_text = f"paraphrase: {example['sentence1']} </s> {example['sentence2']}"
    model_input = tokenizer(input_text, padding="max_length", truncation=True, max_length=512)
    model_input["labels"] = [example["label"]]  # This is a dummy label for minimal training
    return model_input

processed_dataset = dataset.map(preprocess)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./checkpoint",
    per_device_train_batch_size=1,
    max_steps=1,
    save_steps=1,
    save_total_limit=1
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer
)

# Train and save
trainer.train()
trainer.save_model()
