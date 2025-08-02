import gc
import logging
import os

import torch

from transformers import AutoTokenizer
from transformers.models.blt.modeling_blt import BLTForCausalLM


logger = logging.getLogger()

os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"

gc.collect()
torch.cuda.empty_cache()


def main(prompt: str = "my name is", model_name: str = ""):
    model = BLTForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False, use_cache=False)

    output_text = tokenizer.decode(generated_ids[0])

    print(f'Model: "{model_name}"')
    #  print(f'Prompt: "{prompt}"')
    print(f'Completion: "{output_text}"')


if __name__ == "__main__":
    main(model_name="itazap/blt-1b-testing")
