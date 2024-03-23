import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("LLM360/CrystalChat")
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("LLM360/CrystalChat").to(device)

    prompt = "<s> <|sys_start|> You are an AI assistant. You will be given a task. You must generate a detailed and long answer. <|sys_end|> <|im_start|> Write a python function that takes a list of integers and returns the squared sum of the list. <|im_end|>"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_tokens = model.generate(input_ids, do_sample=True, max_length=400)

    print("-" * 20 + "Output for model" + 20 * "-")
    print(tokenizer.batch_decode(gen_tokens)[0])
