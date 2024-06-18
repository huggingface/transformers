import torch

from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import SinkCache

model_id = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
      model_id, low_cpu_mem_usage=True, device_map='auto',
      torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = 'left')
tokenizer.pad_token = tokenizer.eos_token
device = model.device

prefix_list = ["Hello, my name is yy", "What is your name?"]
dialog_history = ""
cache = SinkCache(window_length=20, num_sink_tokens=4)

for prefix in prefix_list:
    dialog_history += prefix
    inputs = tokenizer(dialog_history, return_tensors='pt').to(device)
    input_length = inputs.input_ids.shape[-1]

    gen_out = model.generate(**inputs, do_sample=False, max_new_tokens=64,
                            use_cache=True,
                            past_key_values=cache,
                            pad_token_id=tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                        )

    decoded = tokenizer.decode(gen_out.sequences[0, input_length:], skip_special_tokens=True)
    dialog_history += decoded
    cache = gen_out.past_key_values

    print(decoded)
