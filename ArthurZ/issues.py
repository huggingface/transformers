from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", use_fast = False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens(['<TARGET_BEGIN>', '<TARGET_END>', '<REPR_BEGIN>', '<REPR_END>'], special_tokens=True)
print(tokenizer("<REPR_END>inform"))