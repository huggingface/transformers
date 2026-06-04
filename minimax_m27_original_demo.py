from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

TORCH_USE_CUDA_DSA=1

MODEL_PATH = "MiniMaxAI/MiniMax-M2.7"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

messages = [
    {"role": "user", "content": [{"type": "text", "text": "What is your favourite condiment?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"}]},
    {"role": "user", "content": [{"type": "text", "text": "Do you have mayonnaise recipes?"}]}
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, return_dict=False).to("cuda")

generated_ids = model.generate(model_inputs, max_new_tokens=10, generation_config=model.generation_config)

response = tokenizer.batch_decode(generated_ids)[0]

print(response)
