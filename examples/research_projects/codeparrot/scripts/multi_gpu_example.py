from accelerate import Accelerator


def complete_code_multi_gpu(model, tokenizer, dataloader, num_completions=1, **gen_kwargs):
    accelerator = Accelerator()
    #import ipdb; ipdb.set_trace()                                                                   
    model, dataloader = accelerator.prepare(model, dataloader)
    for step, batch in enumerate(dataloader):
        with torch.no_grad():                                                                        
            generated_tokens = accelerator.unwrap_model(model).generate(input_ids=batch, num_return_sequences=num_completions, **gen_kwargs)
            generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
    code_gens = []                                                                                   
    for s in generated_tokens:                                                                       
        code_gens.append(tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return [first_block(code_gen) for code_gen in code_gens]

