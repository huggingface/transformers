import logging
import os

import torch

from transformers.models.blt.modeling_blt import BLTForCausalLM

from transformers.models.blt.tokenization_blt import BLTTokenizer

logger = logging.getLogger()

os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"

import gc
gc.collect()
torch.cuda.empty_cache()


def main(prompt: str = "my name is", model_name: str = "blt-1b"):
    device = "cuda"

    model = BLTForCausalLM.from_pretrained(
            "itazap/blt-1b"
        ).to(device)

    tokenizer = BLTTokenizer(add_bos_token=True, add_eos_token=True)

    input_ids = torch.tensor([tokenizer.encode(prompt, add_eos=False)]).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=200,
            do_sample=False, 
            temperature=1.0
        )

    generated_ids = output_ids[0][len(input_ids[0]):]
    output_text = tokenizer.decode(generated_ids.tolist())
    
    print(f'Prompt: "{prompt}"')
    print(f'Completion: "{output_text}"')
    print('here')


if __name__ == "__main__":
    main()
