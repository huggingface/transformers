from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def generate(prompt, model, tokenizer, token_healing):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    generation_config = GenerationConfig(
        max_new_tokens=16,
        temperature=0.7, do_sample=True, top_p=0.95, top_k=40,
        pad_token_id=model.config.pad_token_id,
    )
    output = model.generate(
        inputs=input_ids, token_healing=token_healing, generation_config=generation_config
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

model_name_or_path = 'TheBloke/deepseek-llm-7B-base-GPTQ'
completion_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map='auto',
    trust_remote_code=False,
    revision='main',
    use_cache=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

test_prompts = [
    'The link is <a href="http:',
    'The link is <a href="http', # test aggressive healing http->https
    'I read a book about ', # test trailing whitespace
    'I read a book about', # test no-op
    'I', # test single-token
    'An example ["like this"] and another example [',
]
for prompt in test_prompts:
    print(f'\nOriginal prompt:\n{prompt}\n')

    healed_output = generate(prompt, completion_model, tokenizer, token_healing=True)
    print(f'Generation without token healing:\n{healed_output}\n')

    output = generate(prompt, completion_model, tokenizer, token_healing=False)
    print(f'Generation without token healing:\n{output}\n')
