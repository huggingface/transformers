"""
This code tests the logits and intermediary activations to check if importing the module went well.
It is a temporary file meant to be used with the code from commit number 68bbfab60d57f1bff1281e9ea8cdbcc0053f0391
from the branch yaswanth19-add-janus-model in https://github.com/hsilva664/transformers.git. Later, the way the
functions that generate images are called will change
"""

import argparse
import os
import PIL.Image
import torch
import numpy as np
from transformers import JanusForConditionalGeneration, enable_full_determinism

def main(args):
    enable_full_determinism(0)

    # Emu3 example also loads this class like this in tests and in the docs, instead of using Auto
    model = JanusForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = model.eval()
    answers = torch.load(args.answers_file)
    # TODO: These parameters are passed via inference code directly in the Janus codebase, change later
    parallel_size = 4
    temperature = 1
    cfg_weight = 5
    image_token_num_per_image = 576
    img_size = 384
    patch_size = 16

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    with torch.inference_mode():
        for i in range(image_token_num_per_image):
            # Will not use processor and tokenizer for now. Loading output ids directly
            hidden_states = answers["hidden_states_list"][i]

            logits = model.gen_head(hidden_states[:, -1, :].to(device=model.gen_head.device))
            assert torch.isclose(logits.cpu(), answers["logit_list"][i], atol=1e-3).all()
            # Use this to increase precision thereafter
            logits = answers["logit_list"][i].cuda()

            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            assert torch.isclose(next_token.cpu(), answers["next_token_list"][i], atol=1e-3).all()

            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = model.gen_aligner(model.gen_embed(next_token.to(device=model.gen_embed.weight.device))
                                           .to(device=model.gen_aligner.device)
                                           )
            inputs_embeds = img_embeds.unsqueeze(dim=1)

            assert torch.isclose(inputs_embeds.cpu(), answers["inputs_embeds_list"][i], atol=1e-3).all()
            # No need for the following, as llm is being abstracted away
            # inputs_embeds = answers["inputs_embeds_list"][i].cuda()

    dec = model.gen_vision.decode_code(generated_tokens.to(dtype=torch.int, device=model.gen_vision.device),
                                                shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size])
    assert torch.isclose(dec.cpu(), answers["dec"], atol=1e-3).all()
    # No need for the rest of the code, they are all operations that do not depend on the model

if __name__ == '__main__':
    # Create the args with argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        help="Local directory to get pretrained weights from",
                        )
    parser.add_argument("--answers_file",
                        help="Local directory to save the model weights to",
                        )
    args = parser.parse_args()
    main(args)