import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def generate(inputs, model, tokenizer, token_healing):
    input_ids = tokenizer(inputs, return_tensors="pt", padding=True, device_map="auto").input_ids
    generation_config = GenerationConfig(
        max_new_tokens=8,
        token_healing=token_healing,
        pad_token_id=model.config.pad_token_id,
        repetition_penalty=1.1,
    )
    output = model.generate(inputs=input_ids, generation_config=generation_config)
    return tokenizer.batch_decode(output, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--model_name_or_path", type=str, default="TheBloke/deepseek-llm-7B-base-GPTQ")
    args = parser.parse_args()

    prompts = (
        [args.prompt]
        if args.prompt
        else [
            'An example ["like this"] and another example [',
            'The link is <a href="http:',
            'The link is <a href="http',  # test aggressive healing http->https
            "I read a book about ",  # test trailing whitespace
            "I read a book about",  # test nothing to heal
        ]
    )

    model_name_or_path = args.model_name_or_path
    completion_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        use_cache=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    raw_output = generate(prompts, completion_model, tokenizer, token_healing=False)
    healed_output = generate(prompts, completion_model, tokenizer, token_healing=True)

    for p, a, b in zip(prompts, raw_output, healed_output):
        print(f"\nPrompt: {p}\nWithout healing:\n{a}\nWith healing:\n{b}")

    # You can also use token healing in isolation
    # This can be useful if you have other work to do before the generation
    # Or if you want to delegate generation to another process
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.cuda()
    healed_ids = completion_model.heal_tokens(input_ids)
    healed_prompts = tokenizer.batch_decode(healed_ids, skip_special_tokens=True)
    print("\nhealed prompts:")
    for p in healed_prompts:
        print(p)


if __name__ == "__main__":
    main()
