from vllm import LLM, SamplingParams

llm = LLM("openai-community/gpt2", model_impl="transformers", tensor_parallel_size=1)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# from transformers import AutoModelForCausalLM, AutoTokenizer

# tok = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt2")

# inputs = tok("Hello there ", return_tensors="pt")
# outputs = model.generate(**inputs)
# print(tok.batch_decode(outputs))