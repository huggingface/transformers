import json
import multiprocessing
import os
import re

from datasets import load_dataset, load_metric
from tqdm import tqdm

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed


transformers.logging.set_verbosity_error()
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def first_block(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("\nclass|\ndef|\n#|\n@|\nprint|\nif", string)[0].rstrip()


def complete_code(pipe, prompt, num_completions=1, **gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    prompt = pipe.tokenizer.eos_token + prompt
    code_gens = pipe(prompt, num_return_sequences=num_completions, **gen_kwargs)
    return [first_block(code_gen["generated_text"][len(prompt) :]) for code_gen in code_gens]


# Settings
gen_kwargs = {
    "do_sample": True,
    "temperature": 0.2,
    "max_new_tokens": 256,
    "top_p": 0.95,
    "top_k": 0,
}

bs = 10
samples = 200
num_workers = multiprocessing.cpu_count()
model_ckpt = "lvwerra/codeparrot"
set_seed(1)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForCausalLM.from_pretrained(model_ckpt)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Load evaluation dataset and metric
human_eval = load_dataset("openai_humaneval")
code_eval_metric = load_metric("code_eval")

# Generate completions for evaluation set
n_tasks = len(human_eval["test"])
generations, references = [], []
for task in tqdm(range(n_tasks)):
    task_generations = []
    prompt = human_eval["test"][task]["prompt"].strip()
    for batch in range(samples // bs):
        task_generations.extend(complete_code(pipe, prompt, num_completions=bs, **gen_kwargs))
    generations.append([prompt + gen for gen in task_generations])
    test_func = human_eval["test"][task]["test"]
    entry_point = f"check({human_eval['test'][task]['entry_point']})"
    references.append("\n" + test_func + "\n" + entry_point)

# Evaluate completions with "code_eval" metric
pass_at_k, _ = code_eval_metric.compute(references=references, predictions=generations, num_workers=num_workers)
print(f"Results: {pass_at_k}")

# Save results to json file
with open("eval_results.json", "w") as fp:
    json.dump(pass_at_k, fp)
