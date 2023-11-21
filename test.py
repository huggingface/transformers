import transformers
import torch
device = "cuda:0"
model_name = "adept/persimmon-8b-chat"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16, device_map=device, use_flash_attention_2=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)


def generate_response(user_input):
    
    # Format the input using the provided template
    prompt = f"human: {user_input}\n\nadept:"

    # Tokenize and encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate a response
    outputs = model.generate(inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    return response


user_input = "Who is Donald Trump?"
response = generate_response(user_input)
print(response)