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

These are well established decoding methods, and should be your starting point for text generation tasks.

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

We enable custom decoding methods through model repositories, assuming a specific model tag and file structure (see subsection below). This feature is an extension of [custom modeling code](./models.md#custom-models) and, like such, requires setting `trust_remote_code=True`.

If a model repository holds a custom decoding method, the easiest way to try it out is to load the model and generate with it:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

# `transformers-community/custom_generate_example` holds a copy of `Qwen/Qwen2.5-0.5B-Instruct`, but
# with custom generation code -> calling `generate` uses the custom decoding method!
tokenizer = AutoTokenizer.from_pretrained("transformers-community/custom_generate_example")
model = AutoModelForCausalLM.from_pretrained(
    "transformers-community/custom_generate_example", device_map="auto", trust_remote_code=True
)

inputs = tokenizer(["The quick brown"], return_tensors="pt").to(model.device)
# The custom decoding method is a minimal greedy decoding implementation. It also prints a custom message at run time.
gen_out = model.generate(**inputs)
# you should now see its custom message, "✨ using a custom generation method ✨"
print(tokenizer.batch_decode(gen_out, skip_special_tokens=True))
'The quick brown fox jumps over a lazy dog, and the dog is a type of animal. Is'
```

Model repositories with custom decoding methods have a special property: their decoding method can be loaded from **any** model through [`~GenerationMixin.generate`]'s `custom_generate` argument. This means anyone can create and share their custom generation method to potentially work with any Transformers model, without requiring users to install additional Python packages.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")

inputs = tokenizer(["The quick brown"], return_tensors="pt").to(model.device)
# `custom_generate` replaces the original `generate` by the custom decoding method defined in
# `transformers-community/custom_generate_example`
gen_out = model.generate(**inputs, custom_generate="transformers-community/custom_generate_example", trust_remote_code=True)
print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
'The quick brown fox jumps over a lazy dog, and the dog is a type of animal. Is'
```

You should read the `README.md` file of the repository containing the custom generation strategy to see what the new arguments and output type differences are, if they exist. Otherwise, you can assume it works like the base [`~GenerationMixin.generate`] method.

> [!TIP]
> You can find all custom decoding methods by [searching for their custom tag.](https://huggingface.co/models?other=custom_generate), `custom_generate`

Consider the Hub repository [transformers-community/custom_generate_example](https://huggingface.co/transformers-community/custom_generate_example) as an example. The `README.md` states that it has an additional input argument, `left_padding`, which adds a number of padding tokens before the prompt.

```py
gen_out = model.generate(
    **inputs, custom_generate="transformers-community/custom_generate_example", trust_remote_code=True, left_padding=5
)
print(tokenizer.batch_decode(gen_out)[0])
'<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>The quick brown fox jumps over the lazy dog.\n\nThe sentence "The quick'
```

If the custom method has pinned Python requirements that your environment doesn't meet, you'll get an exception about missing requirements. For instance, [transformers-community/custom_generate_bad_requirements](https://huggingface.co/transformers-community/custom_generate_bad_requirements) has an impossible set of requirements defined in its `custom_generate/requirements.txt` file, and you'll see the error message below if you try to run it.

```
ImportError: Missing requirements in your local environment for `transformers-community/custom_generate_bad_requirements`:
foo (installed: None)
bar==0.0.0 (installed: None)
torch>=99.0 (installed: 2.6.0)
```

Updating your Python requirements accordingly will remove this error message.

### Creating a custom decoding method

To create a new decoding method, you need to create a new [**Model**](https://huggingface.co/new) repository and push a few files into it.
1. The model you've designed your decoding method with.
2. `custom_generate/generate.py`, which contains all the logic for your custom decoding method.
3. `custom_generate/requirements.txt`, used to optionally add new Python requirements and/or lock specific versions to correctly use your method.
4. `README.md`, where you should add the `custom_generate` tag and document any new arguments or output type differences of your custom method here.

After you've added all required files, your repository should look like this

```
your_repo/
├── README.md          # include the 'custom_generate' tag
├── config.json
├── ...
└── custom_generate/
    ├── generate.py
    └── requirements.txt
```

#### Adding the base model

The starting point for your custom decoding method is a model repository just like any other. The model to add to this repository should be the model you've designed your method with, and it is meant to be part of a working self-contained model-generate pair. When the model in this repository is loaded, your custom decoding method will override `generate`. Don't worry -- your decoding method can still be loaded with any other Transformers model, as explained in the section above.

If you simply want to copy an existing model, you can do

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("source/model_repo")
model = AutoModelForCausalLM.from_pretrained("source/model_repo")
tokenizer.save_pretrained("your/decoding_method", push_to_hub=True)
model.save_pretrained("your/decoding_method", push_to_hub=True)
```

#### generate.py

This is the core of your decoding method. It *must* contain a method named `generate`, and this method *must* contain a `model` argument as its first argument. `model` is the model instance, which means you have access to all attributes and methods in the model, including the ones defined in [`GenerationMixin`] (like the base `generate` method).

> [!WARNING]
> `generate.py` must be placed in a folder named `custom_generate`, and not at the root level of the repository. The file paths for this feature are hardcoded.

Under the hood, when the base [`~GenerationMixin.generate`] method is called with a `custom_generate` argument, it first checks its Python requirements (if any), then locates the custom `generate` method in `generate.py`, and finally calls the custom `generate`. All received arguments and `model` are forwarded to your custom `generate` method, with the exception of the arguments used to trigger the custom generation (`trust_remote_code` and `custom_generate`).

This means your `generate` can have a mix of original and custom arguments (as well as a different output type) as shown below.

```py
import torch

def generate(model, input_ids, generation_config=None, left_padding=None, **kwargs):
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

Follow the recommended practices below to ensure your custom decoding method works as expected.
- Feel free to reuse the logic for validation and input preparation in the original [`~GenerationMixin.generate`].
- Pin the `transformers` version in the requirements if you use any private method/attribute in `model`.
- Consider adding model validation, input validation, or even a separate test file to help users sanity-check your code in their environment.

Your custom `generate` method can relative import code from the `custom_generate` folder. For example, if you have a `utils.py` file, you can import it like this:

```py
from .utils import some_function
```

Only relative imports from the same-level `custom_generate` folder are supported. Parent/sibling folder imports are not valid. The `custom_generate` argument also works locally with any directory that contains a `custom_generate` structure. This is the recommended workflow for developing your custom decoding method.


#### requirements.txt

You can optionally specify additional Python requirements in a `requirements.txt` file inside the `custom_generate` folder. These are checked at runtime and an exception will be thrown if they're missing, nudging users to update their environment accordingly.

#### README.md

The root level `README.md` in the model repository usually describes the model therein. However, since the focus of the repository is the custom decoding method, we highly recommend to shift its focus towards describing the custom decoding method. In addition to a description of the method, we recommend documenting any input and/or output differences to the original [`~GenerationMixin.generate`]. This way, users can focus on what's new, and rely on Transformers docs for generic implementation details.

For discoverability, we highly recommend you to add the `custom_generate` tag to your repository. To do so, the top of your `README.md` file should look like the example below. After you push the file, you should see the tag in your repository!

```
---
library_name: transformers
tags:
  - custom_generate
---

(your markdown content here)
```

Recommended practices:
- Document input and output differences in [`~GenerationMixin.generate`].
- Add self-contained examples to enable quick experimentation.
- Describe soft-requirements such as if the method only works well with a certain family of models.


## Resources

Read the [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate) blog post for an explanation of how common decoding strategies work.
