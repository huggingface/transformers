import torch
import time # Import the time module
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "/root/.cache/huggingface/hub/gpt-oss-120b/"
device = "cuda:0" # Explicitly target the first GPU

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    # We are explicitly loading to a single device
    device_map=device,
)

model = torch.compile(model)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

isl = 2048

input_ids = torch.randint(0, tokenizer.vocab_size, (1, isl)).to(device)

results = {}

for osl in [1,]:
#for osl in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    print("  -> Running warm-up...")
    _ = model.generate(
        input_ids,
        max_new_tokens=osl,
        do_sample=False,
    )
    torch.cuda.synchronize()
    print("  -> Warm-up complete.")

    print("  -> Running measurement...")
    torch.cuda.synchronize() 
    start_time = time.time()

    outputs = model.generate(
        input_ids,
        max_new_tokens=osl,
        do_sample=False
    )

    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    results[osl] = elapsed_time
    print(f"  -> For {osl} output tokens, generation took: {elapsed_time:.4f} seconds")


print("\n--- Benchmark Summary ---")
for osl, t in results.items():
    print(f"Tokens: {osl:<4} | Time: {t:.4f} sec")
