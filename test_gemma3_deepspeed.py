# test_gemma3_deepspeed.py
import torch
from transformers import (
    Gemma3ForConditionalGeneration,
    Gemma3Config,
    Trainer,
    TrainingArguments,
)
import deepspeed
import json

# Create minimal config
config = Gemma3Config(
    vocab_size=1000,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    max_position_embeddings=1024,
    pad_token_id=0,
)

# DeepSpeed ZeRO-3 config
deepspeed_config = {
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "optimizer": {"type": "AdamW", "params": {"lr": 5e-5}},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
    },
    "fp16": {"enabled": True},
}

# Save config
with open("deepspeed_config.json", "w") as f:
    json.dump(deepspeed_config, f)


def main():
    model = Gemma3ForConditionalGeneration(config)
    training_args = TrainingArguments(
        output_dir="./test_output",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        deepspeed="deepspeed_config.json",
        logging_steps=1,
    )
    dummy_data = [
        {
            "input_ids": torch.randint(0, 1000, (512,)),
            "labels": torch.randint(0, 1000, (512,)),
        }
        for _ in range(4)
    ]

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dummy_data,
    )

    print("Model initialized successfully with DeepSpeed ZeRO-3!")
    return trainer


if __name__ == "__main__":
    trainer = main()
