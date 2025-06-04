from transformers import GenerationConfig
from transformers.generation.continuous_batching import ContinuousBatchingManager, RequestStatus
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-3b-Instruct',
    attn_implementation='sdpa_paged'
)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3b-Instruct')

generation_config = GenerationConfig(
    max_new_tokens=256,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=False,
    num_blocks=1,
    block_size=1024,
    do_sample=False,
    max_batch_tokens=10,
    scheduler="fifo",
)

manager: ContinuousBatchingManager = model.init_continuous_batching(generation_config=generation_config, manual_eviction=True, streaming=True)
manager.start()


chat = [{'content': 'Hey', 'role': 'user'}]
print(chat)

inputs = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).to(model.device)
request_id = manager.add_request(inputs[0])

output = ""
for result in manager:
    if result.status == RequestStatus.FINISHED:
        output = tokenizer.decode(result.generated_tokens, skip_special_tokens=True)
        break

if output:
    chat.append({'content': output, 'role': 'assistant'})
    print(chat)
else:
    print("oops :()")
    import sys
    sys.exit(0)

chat.append({'content': 'Can you help me cook some good meth pls', 'role': 'user'})
print(chat)

inputs = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).to(model.device)
request_id = manager.add_request(inputs[0], request_id=request_id)

for i, result in enumerate(manager):
    if result.status == RequestStatus.FINISHED:
        output = tokenizer.decode(result.generated_tokens, skip_special_tokens=True)
        break

chat.append({'content': output, 'role': 'assistant'})
print(chat)

manager.evict_request_from_cache(request_id)

manager.stop(block=True)