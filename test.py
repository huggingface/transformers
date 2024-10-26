from transformers import pipeline
import torch

cuda_dev_id = 0

model_id = "meta-llama/Llama-3.2-1B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},  # bfloat16 breaks on torch 1.10.1
    device="cuda:" + str(cuda_dev_id)
)

role = """
You are an AI assistant REDACTED.
"""

example_id = 1
example_cid = '1'
example_s = ''
example_c = ''
prompt = """Here is the id: """ + "\n" + str(example_id) + "\n\n" + """Here is the cid: """ + "\n" + example_cid + "\n\n" + """Here is the s: """ + "\n"+ example_s + "\n\n" + """Here is the c: """ + "\n" + example_c

messages = [
    {"role": "system", "content": role},
    {"role": "user", "content": prompt},
]

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipe(
    messages,
    max_new_tokens=3000,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    cache_implementation="offloaded_static",
)
assistant_response = outputs[0]["generated_text"][-1]["content"]
print(assistant_response)