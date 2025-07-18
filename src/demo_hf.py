import logging
import os

import torch

from transformers.models.blt.modeling_blt import BLTForCausalLM

from transformers.models.blt.tokenization_blt import BLTTokenizer

from transformers import AutoTokenizer

logger = logging.getLogger()

os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"

import gc
gc.collect()
torch.cuda.empty_cache()


def main(prompt: str = "my name is", model_name: str = ""):

    model = BLTForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
           # attn_implementation="eager"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False, use_cache=False)

    output_text = tokenizer.decode(generated_ids[0])
    
    print(f'Model: "{model_name}"')
  #  print(f'Prompt: "{prompt}"')
    print(f'Completion: "{output_text}"')


if __name__ == "__main__":
 #   SNAPSHOT_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--itazap--blt-1b/snapshots/bb8c23be2c2f065f0ee315ec2066ac1d3c78722a")
  #  main(model_name=SNAPSHOT_PATH)
    main(model_name="itazap/blt-1b-testing")

