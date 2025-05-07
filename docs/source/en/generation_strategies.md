<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Generation strategies

A decoding strategy informs how a model should select the next generated token. There are many types of decoding strategies, and choosing the appropriate one has a significant impact on the quality of the generated text.

This guide will help you understand the different decoding strategies available in Transformers and how and when to use them.

## Basic decoding methods

These are well established decoding methods, and should be you starting point for text generation tasks.

### Greedy search

Greedy search is the default decoding strategy. It selects the next most likely token at each step. Unless specified in [`GenerationConfig`], this strategy generates a maximum of 20 new tokens.

Greedy search works well for tasks with relatively short outputs where creativity is not a priority. However, it breaks down when generating longer sequences because it begins to repeat itself.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to("cuda")
# explicitly set to default length because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a suite of tools and services for building, deploying, and maintaining natural language processing'
```

### Sampling

Sampling, or multinomial sampling, randomly selects a token based on the probability distribution over the entire model's vocabulary (as opposed to the most likely token, as in greedy search). This means every token with a non-zero probability has a chance to be selected. Sampling strategies reduce repetition and can generate more creative and diverse outputs.

Enable multinomial sampling with `do_sample=True` and `num_beams=1`.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, num_beams=1)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company 🤗\nWe are open-source and believe that open-source is the best way to build technology. Our mission is to make AI accessible to everyone, and we believe that open-source is the best way to achieve that.'
```

### Beam search

Beam search keeps track of several generated sequences (beams) at each time step. After a certain number of steps, it selects the sequence with the highest *overall* probability. Unlike greedy search, this strategy can "look ahead" and pick a sequence with a higher probability overall even if the initial tokens have a lower probability. It is best suited for input-grounded tasks, like describing an image or speech recognition. You can also use `do_sample=True` with beam search to sample at each step, but beam search will still greedily prune out low probability sequences between steps.

> [!TIP]
> Check out the [beam search visualizer](https://huggingface.co/spaces/m-ric/beam_search_visualizer) to see how beam search works.

Enable beam search with the `num_beams` parameter (should be greater than 1 otherwise it's equivalent to greedy search).

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=2)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
"['Hugging Face is an open-source company that develops and maintains the Hugging Face platform, which is a collection of tools and libraries for building and deploying natural language processing (NLP) models. Hugging Face was founded in 2018 by Thomas Wolf']"
```

## Advanced decoding methods

Advanced decoding methods aim at either tackling specific generation quality issues (e.g. repetition) or at improving the generation throughput in certain situations. These techniques are more complex, and may not work correctly with all models.

### Speculative decoding

[Speculative](https://hf.co/papers/2211.17192) or assistive decoding isn't a search or sampling strategy. Instead, speculative decoding adds a second smaller model to generate candidate tokens. The main model verifies the candidate tokens in a single `forward` pass, which speeds up the decoding process overall. This method is especially useful for LLMs where it can be more costly and slower to generate tokens. Refer to the [speculative decoding](./llm_optims#speculative-decoding) guide to learn more.

Currently, only greedy search and multinomial sampling are supported with speculative decoding. Batched inputs aren't supported either.

Enable speculative decoding with the `assistant_model` parameter. You'll notice the fastest speed up with an assistant model that is much smaller than the main model. Add `do_sample=True` to enable token validation with resampling.

<hfoptions id="spec-decoding">
<hfoption id="greedy search">

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine'
```

Speculative decoding is also supported in [`Pipeline`] with the `assistant_model` parameter.

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B",
    assistant_model="meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16
)
pipe_output = pipe("Once upon a time, ", max_new_tokens=50, do_sample=False)
pipe_output[0]["generated_text"]
```

</hfoption>
<hfoption id="multinomial sampling">

Add the `temperature` parameter to control sampling randomness. For speculative decoding, a lower temperature may improve latency.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that is dedicated to creating a better world through technology.'
```

</hfoption>
</hfoptions>

#### Prompt lookup decoding

[Prompt lookup decoding](./llm_optims#prompt-lookup-decoding) is a variant of speculative decoding that uses overlapping n-grams as the candidate tokens. It works well for input-grounded tasks such as summarization. Refer to the [prompt lookup decoding](./llm_optims#prompt-lookup-decoding) guide to learn more.

Enable prompt lookup decoding with the `prompt_lookup_num_tokens` parameter.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", torch_dtype=torch.float16).to("cuda")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", torch_dtype=torch.float16).to("cuda")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, assistant_model=assistant_model, max_new_tokens=20, prompt_lookup_num_tokens=5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine learning models. It offers a variety of tools'
```

### Self-speculative decoding

Early exiting uses the earlier hidden states from the language modeling head as inputs, effectively skipping layers to yield a lower quality output. The lower quality output is used as the assistant output and self-speculation is applied to fix the output using the remaining layers. The final generated result from this self-speculative method is the same (or has the same distribution) as the original models generation.

The assistant model is also part of the target model, so the caches and weights can be shared, resulting in lower memory requirements.

For a model trained with early exit, pass `assistant_early_exit` to [`~GenerationMixin.generate`].

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = "Alice and Bob"
checkpoint = "facebook/layerskip-llama3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(checkpoint)
outputs = model.generate(**inputs, assistant_early_exit=4, do_sample=False, max_new_tokens=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

#### Universal assisted decoding

Universal assisted decoding (UAD) enables the main and assistant models to use different tokenizers. The main models input tokens are re-encoded into assistant model tokens. Candidate tokens are generated in the assistant encoding which are re-encoded into the main model candidate tokens. The candidate tokens are verified as explained in [speculative decoding](#speculative-decoding).

Re-encoding involves decoding token ids into text and encoding the text with a different tokenizer. To prevent tokenization discrepancies during re-encoding, UAD finds the longest common sub-sequence between the source and target encodings to ensure the new tokens include the correct prompt suffix.

Add the `tokenizer` and `assistant_tokenizer` parameters to [`~GenerationMixin.generate`] to enable UAD.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = "Alice and Bob"

assistant_tokenizer = AutoTokenizer.from_pretrained("double7/vicuna-68m")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
inputs = tokenizer(prompt, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b")
assistant_model = AutoModelForCausalLM.from_pretrained("double7/vicuna-68m")
outputs = model.generate(**inputs, assistant_model=assistant_model, tokenizer=tokenizer, assistant_tokenizer=assistant_tokenizer)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']
```

### Contrastive search

[Contrastive search](https://huggingface.co/papers/2202.06417) is a decoding strategy that aims to reduce repetition even while generating longer sequences. This strategy compares how similar a generated token is against previous tokens, and if they're more similar, a penalty is applied.

Enable contrastive search with the `penalty_alpha` and `top_k` parameters. The `penalty_alpha` manages the penalty applied and `top_k` is the number of most likely tokens to return.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=100, penalty_alpha=0.6, top_k=4)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for building and deploying AI models.\nHugging Face is an open-source company that provides a platform for building and deploying AI models. The platform allows developers to build and deploy AI models, as well as collaborate with other developers.\nHugging Face was founded in 2019 by Thibault Wittemberg and Clément Delangue. The company is based in Paris, France.\nHugging Face has'
```

### DoLa

[Decoding by Contrasting Layers (DoLa)](https://hf.co/papers/2309.03883) is a contrastive decoding strategy for improving factuality and reducing hallucination. This strategy works by contrasting the logit differences between the final and early layers. As a result, factual knowledge localized to particular layers are amplified. DoLa is not recommended for smaller models like GPT-2.

Enable DoLa with the following parameters.

- `dola_layers` are the candidate layers to be contrasted with the final layer. It can be a string (`low` or `high`) to contrast the lower or higher parts of a layer. `high` is recommended for short-answer tasks like TruthfulQA. `low` is recommended for long-answer reasoning tasks like GSM8K, StrategyQA, FACTOR, and VicunaQA.

  When a model has tied word embeddings, layer 0 is skipped and it begins from layer 2.

  It can also be a list of integers that represent the layer indices between 0 and the total number of layers. Layer 0 is the word embedding, 1 is the first transformer layer, and so on. Refer to the table below for the range of layer indices depending on the number of model layers.

  | layers | low | high |
  |---|---|---|
  | > 40 | (0, 20, 2) | (N - 20, N, 2) |
  | <= 40 | range(0, N // 2, 2) | range(N // 2, N, 2) |

- `repetition_penalty` reduces repetition and it is recommended to set it to 1.2.

<hfoptions id="dola">
<hfoption id="contrast higher layers">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", torch_dtype=torch.float16).to("cuda")
inputs = tokenizer("What is the highest peak in the world??", return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=50, dola_layers="high", do_sample=False)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
" Mount EverestMount Everest, called Himalaya in Nepali, is the world's highest peak, lying almost 9.5 kilometers above the sea level and the tallest mountain from 19,036.91 ft. The mountain was"
```

</hfoption>
<hfoption id="contrast specific layers">

Contrast layers 18 and 20 with the final layer.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", torch_dtype=torch.float16).to("cuda")
inputs = tokenizer("What is the highest peak in the world?", return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=50, dola_layers=[18,20], do_sample=False, repetition_penalty=1.2)
tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
" Mount EverestMount Everest, called Himalaya in Nepali, is the world's highest peak above sea level and it rises to an incredible height of 29,028 feet above the ocean. Its summit is over a mile taller than Mt"
```

</hfoption>
</hfoptions>

### Diverse beam search

[Diverse beam search](https://hf.co/papers/1610.02424) is a variant of beam search that produces more diverse output candidates to choose from. This strategy measures the dissimilarity of sequences and a penalty is applied if sequences are too similar. To avoid high computation costs, the number of beams is divided into groups.

Enable diverse beam search with the `num_beams`, `num_beam_groups` and `diversity_penalty` parameters (the `num_beams` parameter should be divisible by `num_beam_groups`).

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=6, num_beam_groups=3, diversity_penalty=1.0, do_sample=False)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company 🤗\nWe are an open-source company. Our mission is to democratize AI and make it accessible to everyone. We believe that AI should be used for the benefit of humanity, not for the benefit of a'
```


## Custom decoding methods

Custom decoding methods enable specialized generation behavior such as the following:
- have the model continue thinking if it is uncertain;
- roll back generation if the model gets stuck;
- handle special tokens with custom logic;
- enhanced input preparation for advanced models;

To that end, we enable `generate` to load custom generation recipes from the Hub through its `recipe` argument. This means anyone can create and share their own advanced generation method which, from a user perspective, requires no additional python packages.

<!-- TODO before merging: 1) better repo name (use a `generate-community` org?) 2) prettify the repo -->
```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")

inputs = tokenizer(["The quick brown"], return_tensors="pt").to(model.device)
gen_out = model.generate(**inputs, recipe="joaogante/test_generate_from_hub")  # minimal greedy decoding
print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
'The quick brown fox jumps over a lazy dog, and the dog is a type of animal. Is'
```

### Using a custom decoding method

After you've identified a custom generation recipe on the Hub that you'd like to try, the first thing you should do is to read its README page. When creating a recipe (see below), we recommend their creators to document new arguments and output type differences, if any of these exist. Reasonably, you should assume everything works like the base `generate` unless documented.

Let's consider the Hub repository [`joaogante/test_generate_from_hub`](https://huggingface.co/joaogante/test_generate_from_hub). We can see in its README that it has an additional input argument, `left_padding`, which is an optional integer that will add that number of pad tokens before the prompt. Let's try it out!

```py
gen_out = model.generate(**inputs, recipe="joaogante/test_generate_from_hub", left_padding=5)
print(tokenizer.batch_decode(gen_out)[0])
'<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>The quick brown fox jumps over the lazy dog.\n\nThe sentence "The quick'
```

If the recipe has pinned python requirements and your environment doesn't fulfill them, you'll get an exception about missing requirements. For instance, [`joaogante/test_generate_from_hub_2`](https://huggingface.co/joaogante/test_generate_from_hub_2) has an impossible set of requirements defined in its `requirements.txt`, and you'll see something like this if you try to run it:

```
ValueError: Missing requirements for joaogante/test_generate_from_hub_2:
foo (installed: None)
bar==0.0.0 (installed: None)
torch>=99.0 (installed: 2.6.0+cu126)
```

Updating your python requirements accordingly will remove this error message.

### Creating a custom decoding method

To create a new decoding method, you need to create a new "Model" repository and push a few files into it:
1. (required) `generate.py`, which contains all the logic for your custom decoding method.
2. (optional) `README.md`, the front page on the Hub to your custom generation method;
3. (optional) `requirements.txt`, to add new python requirements and/or lock specific versions for correct utilization of the technique.

Let's dive at each file, and have a look at their requirements and best practices.

#### generate.py

This is the core of your decoding method. It *must* contain a `generate` method, and this method *must* contain a `model` argument as its first argument. `model` is the model instance, which means you have access to all attributes and methods in the model, including the ones defined in `GenerationMixin` (like the original `generate` method).

Under the hood, when the original `generate` method is called with a `recipe` argument, it will first check its requirements (if they exist, see below), then locate the custom `generate` method in `generate.py`, and finally calls the custom `generate`, forwarding all received arguments to it (plus `model`). This means your `generate` can have a mix of orginal and custom arguments, as well as a different output type -- the world is your oyster!

Here is an example of a `generate.py` file that mixes original and custom arguments:

```py
import torch

def generate(model, input_ids, generation_config, left_padding=None, **kwargs):
    generation_config = generation_config or model.generation_config  # default to the model generation config
    cur_length = input_ids.shape[1]
    max_length = generation_config.max_length or cur_length + generation_config.max_new_tokens

    # Example of custom argument: add `left_padding` (integer) pad tokens before the prompt
    if left_padding is not None:
        if not isinstance(left_padding, int) or left_padding < 0:
            raise ValueError(f"left_padding must be an integer larger than 0, but is {left_padding}")

        pad_token = kwargs.pop("pad_token", None) or generation_config.pad_token_id or model.config.pad_token_id
        if pad_token is None:
            raise ValueError("pad_token is not defined")
        batch_size = input_ids.shape[0]
        pad_tensor = torch.full(size=(batch_size, left_padding), fill_value=pad_token).to(input_ids.device)
        input_ids = torch.cat((pad_tensor, input_ids), dim=1)
        cur_length = input_ids.shape[1]

    # Simple greedy decoding loop
    while cur_length < max_length:
        logits = model(input_ids).logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=-1)
        cur_length += 1

    return input_ids
```

Recommended practices:
- The original `generate` has extensive logic for validation and input preparation. Feel free to reuse them;
- If you use any private method/attribute in `model`, pin the `transformers` version you're using in the requirements;
- Consider adding model validation, input validation, or even a separate test file to help users sanity-check your code in their environment.

#### README.md

As the front page of your decoding method, this is where other users get to know it. In addition to a description of the method, we heavily recommend documenting any input and/or output differences to the original `generate`. That way, users can focus on what's new, and rely on `transformers` docs for generic implementation details.

Recommended pratices:
- Document input and output differences in `generate`;
- Add self-contained examples to enable quick experimentation;
- Describe additional soft-requirements (e.g. if the technique only works well with a certain family of models).

#### requirements.txt

If your decoding method has python requirements that are not present in the original `transformers` package, you can specify then in a `requirements.txt` file (just like you would in a python project). These will be checked at runtime, and an exception will be thrown if they are not honored.

## Resources

Read the [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate) blog post for an explanation of how common decoding strategies work.
