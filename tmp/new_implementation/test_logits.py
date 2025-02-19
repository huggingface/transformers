"""
This code tests the logits and intermediary activations to check if importing the module went well.
It is a temporary file meant to be used with the code from commit number 68bbfab60d57f1bff1281e9ea8cdbcc0053f0391
from the branch yaswanth19-add-janus-model in https://github.com/hsilva664/transformers.git. Later, the way the
functions that generate images are called will change
"""

import argparse
import gc
import os
import PIL.Image
import torch
import numpy as np
from transformers import enable_full_determinism, JanusForConditionalGeneration, JanusConfig


def main(args):
    enable_full_determinism(0)

    # Emu3 example also loads this class like this in tests and in the docs, instead of using Auto
    model = JanusForConditionalGeneration(JanusConfig(vq_config={"resolution": 384}))
    # bfloat16 has a problem regarding the cuda implementation of triu on my local device (this does not happen when
    # running on remote multi-gpu)
    # model = model.eval().to(dtype=torch.bfloat16, device="cuda")
    model = model.eval().to(device="cuda")
    d = torch.load(args.model_path)
    # delete the key vqmodel.quantize.codebook_used
    del d["vqmodel.quantize.codebook_used"]
    model.load_state_dict(d)
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

            logits = model.gen_head(hidden_states[:, -1, :].cuda())
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
            img_embeds = model.gen_aligner(model.gen_embed(next_token)

                                           )
            inputs_embeds = img_embeds.unsqueeze(dim=1)

            assert torch.isclose(inputs_embeds.cpu(), answers["inputs_embeds_list"][i], atol=1e-3).all()
            # No need for the following, as llm is being abstracted away
            # inputs_embeds = answers["inputs_embeds_list"][i].cuda()


    del model.vision_model
    del model.aligner
    del model.gen_aligner
    del model.gen_embed
    del model.gen_head
    del model.language_model
    gc.collect()
    torch.cuda.empty_cache()

    # decoding normally is not matching, see intermediate steps
    # dec = model.vqmodel.decode(generated_tokens.to(dtype=torch.int))
    # assert torch.isclose(dec.cpu(), answers["dec"], atol=1e-3).all()

    codebook_entry = model.vqmodel.quantize.get_codebook_entry(generated_tokens.to(dtype=torch.int))
    assert torch.isclose(codebook_entry.cpu(), answers["dec_quant_b"], atol=1e-3).all()
    hidden_states = model.vqmodel.post_quant_conv(codebook_entry)
    assert torch.isclose(hidden_states.cpu(), answers["dec_quant"], atol=1e-3).all()
    pixel_values = model.vqmodel.decoder(answers["dec_quant"].cuda())
    # This does not match. I believe it is due to different cuda versions between my docker container to run Janus
    # and the env in which I was developing transformers
    assert torch.isclose(pixel_values.cpu(), answers["dec_dec"], atol=1e-3).all()

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