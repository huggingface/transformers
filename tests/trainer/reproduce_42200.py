# Swapping to google collab
# Install transformers, Uncomment line below to recreate the problem.
# !pip install transformers torch datasets -q

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, tokenizer, size=900, seq_len=16):
        self.input_ids = torch.randint(0, tokenizer.vocab_size, (size, seq_len))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.input_ids[idx]}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print("logits type: " + str(type(logits)))
    if isinstance(logits, tuple):
        print("BUG CONFIRMED: logits is a tuple of " + str(len(logits)) + " elements")
        print("This is what causes the OOM in issue 42200")
    else:
        print("logits shape: " + str(logits.shape))
        print("Bug not present - logits is a proper array")
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float((predictions == labels).mean())}

# Load Qwen3-0.6B
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

train_dataset = DummyDataset(tokenizer, size=10,  seq_len=16)
eval_dataset  = DummyDataset(tokenizer, size=900, seq_len=16)

args = TrainingArguments(
    output_dir="./tmp_reproduce",
    eval_on_start=True,
    max_steps=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    report_to="none",
    fp16=True,  # use half precision to save VRAM
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    # preprocess_logits_for_metrics intentionally NOT provided
)

print("Starting...")
trainer.train()
print("Done.")