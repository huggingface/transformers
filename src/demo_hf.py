import logging
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from transformers.models.blt_wip.modeling_blt_dev import BLTForCausalLM
from transformers.models.blt_wip.modeling_blt import BLTModel

from transformers.models.blt_wip.tokenization_blt import BLTTokenizer


logger = logging.getLogger()

os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"

import gc
gc.collect()
torch.cuda.empty_cache()


def main(prompt: str = "my name is", model_name: str = "blt-1b"):
    device = "cuda"

    blt_model = BLTModel.from_pretrained("itazap/blt-1b-converted")

    causal_lm = BLTForCausalLM(blt_model.config)
    causal_lm.model.load_state_dict(blt_model.state_dict(), strict=False)
    causal_lm.lm_head.weight = blt_model.local_decoder.lm_head.weight
    causal_lm.save_pretrained( "./blt-1b-causallm")

    # TRUE causal_lm.lm_head.weight == blt_model.local_decoder.lm_head.weight

    model = BLTForCausalLM.from_pretrained("./blt-1b-causallm").to(device)
    
    # FALSE model.lm_head.weight != blt_model.local_decoder.lm_head.weight

    tokenizer = BLTTokenizer(add_bos_token=True, add_eos_token=True)

    input_ids = torch.tensor([tokenizer.encode(prompt, add_eos=False)]).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=200,
            do_sample=False, 
            temperature=1.0,
            pad_token_id=tokenizer.pad_id,
            eos_token_id=tokenizer.eos_id,
        )

    generated_ids = output_ids[0][len(input_ids[0]):]
    output_text = tokenizer.decode(generated_ids.tolist())
    
    print(f'Prompt: "{prompt}"')
    print(f'Completion: "{output_text}"')
    print('here')


if __name__ == "__main__":
    main()
