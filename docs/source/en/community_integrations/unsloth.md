<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Unsloth

[Unsloth](https://unsloth.ai/docs) is a fine-tuning and reinforcement framework that speeds up training and reduces memory usage for large language models. It supports training in 4-bit, 8-bit, and 16-bit precision with custom RoPE and Triton kernels. Unsloth works with Llama, Mistral, Gemma, Qwen, and other model families.

```py
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

dataset = load_dataset("trl-lib/Capybara", split="train[:500]")
dataset = dataset.map(lambda x: {"text": x["conversations"][0]["value"]})

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=2,
        num_train_epochs=1,
    ),
)

trainer.train()
```

## Transformers integration

Unsloth wraps Transformers APIs and patches internal methods for speed.

- `FastLanguageModel.from_pretrained` loads models with [`AutoConfig.from_pretrained`] and [`AutoModelForCausalLM.from_pretrained`]. Before loading, Unsloth patches attention, decoder layer, and rotary embedding classes inside a Transformers model.

- `UnslothTrainer` extends TRL's [`~trl.SFTTrainer`]. Unsloth patches [`~Trainer.compute_loss`] and [`~Trainer.training_step`] to fix gradient accumulation in older Transformers versions.

## Resources

- [Unsloth](https://unsloth.ai/docs) docs
- [Make LLM Fine-tuning 2x faster with Unsloth and TRL](https://huggingface.co/blog/unsloth-trl) blog post