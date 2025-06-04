import datasets
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


torch.set_float32_matmul_precision("high")

model_id = "meta-llama/Llama-3.2-3b-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, attn_implementation="sdpa_paged", torch_dtype=torch.bfloat16, device_map=0
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

generation_config = GenerationConfig(
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=False,
    num_blocks=2048,
    block_size=128,
    do_sample=True,
    max_batch_tokens=1024,  # Maximum number of tokens to process in a single batch
    scheduler="prefill_first",
)

train_dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")

def tokenize_function(examples):
    return tokenizer(examples["question"])

tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]

batch_outputs = model.generate_batch(
    inputs=simple_batch_inputs,
    generation_config=generation_config,
    progress_bar=False,
    enable_visualizer=True,
    tokenizer=tokenizer,
)
