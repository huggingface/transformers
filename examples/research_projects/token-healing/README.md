<!-- back to top link -->
<a name="readme-top"></a>

<!-- ABOUT THE PROJECT -->
## What is token healing?

Token healing rectifies the token boundary bias in greedy tokenization. It does this by trimming and regrowing the prompt to better align with the model's tokenizer, thus enhancing generation quality. The improvement is clearest with completion models.

Example: given a completion prompt with a partial url ending with `:`, the model might have seen the expected completion `://` as a _single_ token in training. However, the prompt's tail token `:` tells it that the next token is not `//`, and so it looks for wrong completions. Such errors compound in auto-regressive language models.

Debiasing token boundaries also addresses output sensitivity to prompts ending with whitespace.

A more thorough explanation can be found on [The Art of Prompt Design: Prompt Boundaries and Token Healing | by Scott Lundberg](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38).

## Usage

```py
prompt = 'The link is <a href="http:'
raw_output = generate(prompt, completion_model, tokenizer, token_healing=False)
# The link is <a href="http:&#47;&#47;www&#47;dailymail&#

# The model saw '://' as a single token in training. Seeing a prompt ending with `:` tells it that the
# next token is likely not `//`, because otherwise it would've seen `://`.
# Thus, it completes with a token other than `//`, in this case, `&`.

healed_output = generate(prompt, completion_model, tokenizer, token_healing=True)
# The link is <a href="http://www.365doki.com/post/3699

# You can also use token healing in isolation
# This can be useful if you have other work to do before the generation
# Or if you want to delegate generation to another process
input_ids = tokenizer(test_prompts, return_tensors='pt', padding=True).input_ids.cuda()
healed_ids = model.heal_tokens(input_ids)
healed_prompts = tokenizer.batch_decode(healed_ids, skip_special_tokens=True)
# outputs the healed prompts without further completion/generation
```

See `run_token_healing.py` for the full example.

<p align="right">(<a href="#readme-top">back to top</a>)</p>