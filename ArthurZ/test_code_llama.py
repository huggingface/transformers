import time

from transformers import pipeline, CodeLlamaTokenizerFast, LlamaForCausalLM, AutoModelForCausalLM
import torch







import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def run(model_name_or_path: str, pad_token_mode: str = "eos_token"):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,low_cpu_mem_usage=True,torch_dtype=torch.float16,device_map="auto",use_cache=True, pad_token_id = 3)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,use_fast=False,legacy=False,padding_side="left")
    if pad_token_mode == "eos_token":
        tokenizer.pad_token = tokenizer.eos_token
    elif pad_token_mode == "unk_token":
        tokenizer.pad_token = tokenizer.unk_token
    elif pad_token_mode == "new_token":
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    else:
        raise NotImplementedError

    print(f"pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # examples extracted from MMLU's abstract_algebra subject with 5-shots
    examples = [
        "The following are multiple choice questions (with answers) about abstract algebra.\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\nStatement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: B\n\nStatement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: C\n\nStatement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: A\n\nFind the characteristic of the ring 2Z.\nA. 0\nB. 3\nC. 12\nD. 30\nAnswer: A\n\nFind the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:",
        "The following are multiple choice questions (with answers) about abstract algebra.\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\nStatement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: B\n\nStatement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: C\n\nStatement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: A\n\nFind the characteristic of the ring 2Z.\nA. 0\nB. 3\nC. 12\nD. 30\nAnswer: A\n\nLet p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.\nA. 8\nB. 2\nC. 24\nD. 120\nAnswer:",
        "The following are multiple choice questions (with answers) about abstract algebra.\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\nStatement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: B\n\nStatement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: C\n\nStatement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: A\n\nFind the characteristic of the ring 2Z.\nA. 0\nB. 3\nC. 12\nD. 30\nAnswer: A\n\nFind all zeros in the indicated finite field of the given polynomial with coefficients in that field. x^5 + 3x^3 + x^2 + 2x in Z_5\nA. 0\nB. 1\nC. 0,1\nD. 0,4\nAnswer:",
        "The following are multiple choice questions (with answers) about abstract algebra.\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\nStatement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: B\n\nStatement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: C\n\nStatement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: A\n\nFind the characteristic of the ring 2Z.\nA. 0\nB. 3\nC. 12\nD. 30\nAnswer: A\n\nStatement 1 | A factor group of a non-Abelian group is non-Abelian. Statement 2 | If K is a normal subgroup of H and H is a normal subgroup of G, then K is a normal subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer:",
    ]
    print("model.generation_config.max_length:", model.generation_config.max_length)
    print("model.generation_config.max_new_tokens:", model.generation_config.max_new_tokens)
    print([len(ex) for ex in examples])

    expected_answers = []
    for example in examples:
        inputs = tokenizer(example, return_tensors="pt",add_special_tokens=False,padding=True,return_token_type_ids=False).to("cuda")
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=1)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        new_text = generated_text[len(example) :]
        expected_answers.append(new_text)

    inputs = tokenizer(examples,return_tensors="pt",add_special_tokens=False,padding=True,return_token_type_ids=False).to("cuda")
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=1)
    batched_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batched_answers = [pred[len(example) :] for pred, example in zip(batched_preds, examples)]

    # when pad_token_mode = "eos_token", the expected_answers and batched_answers should be the same
    # however only the longest example (without padding) is the same, the rest are different
    for i, (expected_answer, batched_answer) in enumerate(zip(expected_answers, batched_answers)):
        if expected_answer == batched_answer:
            print(f"sample {i} is the same")
        else:
            print(
                f"sample {i} is different, expected_answer: `{expected_answer}`, batched_answer: `{batched_answer}`"
            )



model_name_or_path = "meta-llama/Llama-2-7b-hf"
run(model_name_or_path, "eos_token")
run(model_name_or_path, "unk_token")
run(model_name_or_path, "new_token")





prompts = ["""\
import socket

def ping_exponential_backoff(host: str):""",
        """\
import argparse

def main(string: str):
    print(string)
    print(string[::-1])

if __name__ == "__main__":"""]


tokenizer = CodeLlamaTokenizerFast.from_pretrained("codellama/CodeLlama-7b-hf", pad_token = "<s>")
start = time.time()
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
print(f"Loaded LlamaForCausalLM in {time.time() - start}")
model = model.half().cuda()

start = time.time()
inputs = tokenizer(prompts, return_tensors = "pt", padding = True).to("cuda")
outputs = model.generate(**inputs,max_length = 256, temperature = 0.2, do_sample = True, top_p=0.9)
print(f"Generation in {time.time() - start}")

model = model.to(torch.bfloat16)
start = time.time()
inputs = tokenizer(prompts, return_tensors = "pt", padding = True).to("cuda")
outputs = model.generate(**inputs,max_length = 256, temperature = 0.2, do_sample = True, top_p=0.9)
print(f"Generation in bfloat16 {time.time() - start}")


start = time.time()
generator = pipeline("text-generation",model="codellama/CodeLlama-7b-hf",device_map="cuda", torch_dtype = torch.float16)
print(f"Loaded in {time.time() - start}")

start = time.time()
result = generator(prompts, max_length = 256, temperature = 0.2, do_sample = True, top_p=0.9)
print(f"float16 : {time.time() - start}")

generator.model = generator.model.to(torch.bfloat16)
start = time.time()
result = generator(prompts, max_length = 256, temperature = 0.2, do_sample = True, top_p=0.9)
print(f"bfloat16 : {time.time() - start}")

generator.model = generator.model.to(torch.float16)
start = time.time()
result = generator(prompts, max_length = 256, temperature = 0.2, do_sample = True, top_p=0.9)
print(f"float16 : {time.time() - start}")
exit(0)

start = time.time()
result = generator('# Implement quicksort\ndef quicksort(array, left, right):', max_new_tokens = 128, do_sample = False, top_p=0.9)
print(f"float16 : {time.time() - start}")

# print(result[0]["generated_text"])

generator = pipeline("text-generation",model="codellama/CodeLlama-7b-hf", torch_dtype=torch.bfloat16, device_map="auto")
start = time.time()
result = generator('# Implement quicksort\ndef quicksort(array, left, right):', max_new_tokens = 128)
print(f"bfloat16 : {time.time() - start}")
# float16 : 8.526987314224243
# print(result[0]["generated_text"])


generator = pipeline("text-generation",model="codellama/CodeLlama-7b-hf",torch_dtype=torch.float16, device_map="auto")

start = time.time()
result = generator('# Implement quicksort\ndef quicksort(array, left, right):', max_new_tokens = 256)
print(f"float16 : {time.time() - start}")

generator.model.to(torch.bfloat16)

start = time.time()
result = generator('# Implement quicksort\ndef quicksort(array, left, right):', max_new_tokens = 256)
print(f"bfloat16 : {time.time() - start}")
exit(0)
"""
# Implement quicksort
def quicksort(array, left, right):
    if left < right:
        pivot = partition(array, left, right)
        quicksort(array, left, pivot - 1)
        quicksort(array, pivot + 1, right)


# Implement partition
def partition(array, left, right):
    pivot = array[right]
    i = left - 1
    for j in range(left, right):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
"""
generator = pipeline("text-generation",model="codellama/CodeLlama-13b-hf",torch_dtype=torch.float16, device_map="auto")
result = generator('# Implement quickdort\ndef quicksort(array, left, right):', max_new_tokens = 128)

generator = pipeline("text-generation",model="codellama/CodeLlama-34b-hf",torch_dtype=torch.float16, device_map="auto")
result = generator('# Implement quickdort\ndef quicksort(array, left, right):', max_new_tokens = 128)

