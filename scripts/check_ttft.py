from itertools import chain
from time import time

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
dtype = "bfloat16"
amp_enabled = dtype != "float32"
amp_dtype = getattr(torch, dtype)
dtype = torch.bfloat16

print("Loading model to CPU...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).eval()

model.config.token_latency = True
tokenizer = AutoTokenizer.from_pretrained(model_id)
batch_size = 1
num_beams = 1
generate_kwargs = {"do_sample": False, "temperature": 0.9, "num_beams": num_beams}
prompt = "Hi,"

inputs = tokenizer(prompt, return_tensors="pt")
input_len = inputs.input_ids.shape[1]


def run_cpu_inference(num_iter=2):
    print("Starting Warmup (1 iteration)...")
    # Warmup
    model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print("Starting Inference...")
    total_time = 0.0
    total_list = []
    with torch.no_grad(), torch.inference_mode(), torch.cpu.amp.autocast(enabled=amp_enabled):
        for i in range(num_iter):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            tic = time()
            output = model.generate(input_ids, max_new_tokens=10, **generate_kwargs)
            gen_ids = output[0]
            gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            toc = time()

            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in gen_ids]
            total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]

            print(gen_text, total_new_tokens, flush=True)
            print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)

            total_time += toc - tic
            total_list.append(output[1])

    # Results
    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / num_iter
    print("Inference latency: %.5f seconds." % latency)
    first_latency = np.mean([x[0] for x in total_list]) * 1000
    average_2n = list(chain(*[x[1:] for x in total_list]))
    average_2n.sort()
    average_2n_latency = np.mean(average_2n) * 1000
    p90_latency = average_2n[int(len(average_2n) * 0.9)] * 1000
    p99_latency = average_2n[int(len(average_2n) * 0.99)] * 1000
    print("First token average latency: %.2f ms." % first_latency)
    print("Average 2... latency: %.2f ms." % average_2n_latency)
    print("P90 2... latency: %.2f ms." % p90_latency)
    print("P99 2... latency: %.2f ms." % p99_latency)


run_cpu_inference()
