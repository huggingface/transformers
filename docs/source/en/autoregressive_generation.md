<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# Autoregressive Generation

[[open-in-colab]]

Autoregressive generation is the inference-time procedure of iterativelly calling a model with its own generated outputs, given a few initial inputs. This procedure, well explained in [our blog post](https://huggingface.co/blog/how-to-generate), is used with several tasks in different modalities, including:
* [Causal language modeling](tasks/masked_language_modeling)
* [Translation](tasks/translation)
* [Summarization](tasks/summarization)
* [Automatic speech recognition](tasks/asr)
* [Text to speech](tasks/text-to-speech)
* [Image captioning](tasks/image_captioning)

Despite the glaring task differences, autoregressive generation in 🤗 `transformers` shares the same core principles and interface across use cases.

This guide will show you how to:

* Use your model with autoregressive generation
* Avoid common pitfalls
* Take the most of your generative model

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers bitsandbytes>=0.39.0 -q
```


## Generation with LLMs

Let's start with the original and most popular use case of autoregressive generation with transformers: language models. A language model trained on the [causal language modeling task](tasks/masked_language_modeling) will take a sequence of text tokens as input, and returns the probability distribution for the next token. Here's how your LLM forward pass looks like:

<!-- [GIF 1 -- FWD PASS] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_1_1080p.mov"
    ></video>
</figure>

A critical ingredient of autoregressive generation with LLMs is selecting the next token from this probability distribution. Anything goes in this step, as long as you end up with a token selected for the next iteration. This means it can be as simple as selecting the most likely token from the probability distribution, or as complex as applying a dozen transformations before sampling from the resulting distribution. Visually, it looks like this:

<!-- [GIF 2 -- TEXT GENERATION] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_2_1080p.mov"
    ></video>
</figure>

The process depicted above is repeated iterativelly until some stopping criteria is reached. Ideally, this stopping condition is dictated by the model, which should learn when to output an end-of-sequence (EOS) token. When this doesn't happen, generation stops when some pre-defined maximum length is reached.

Properly setting up the token selection step and the stopping criteria is essential to make your model behave as you'd expect on your task. That is why we have a [`~generation.GenerationConfig`] file associated with each model, which contains a good default generative parameterization and is loaded alongside your model.

Let's talk code! If you're interested in basic usage of an LLM, using our high-level [pipeline](pipeline_tutorial) interface is a candidate starting point. However, LLMs often require advanced features like quantization and fine control of the token selection step, which is best done through our [`~generation.GenerationMixin.generate`]. Autoregressive generation with LLMs is also resource-intensive, and should be executed in a GPU for adequate throughput.

<!-- TODO: update example to llama 2 (or a newer popular baseline) when it becomes ungated -->
First, you need to load the model.

```py
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained(
...     "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
... )
```

Note the two flags in the `from_pretrained` call, `device_map` and `load_in_4bit`. The former ensures the model is moved to your GPU(s), while the later applies [4-bit dynamic quantization](main_classes/quantization) to massivelly reduce the resource requirements. There are other ways to initialize the model (more on that later), but this a good baseline to experiment with an LLM.

Next, you need to preprocess your text input with a [tokenizer](tokenizer_summary).

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
>>> model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
```

The `model_inputs` variable holds the tokenized text input, as well as the attention mask. While [`~generation.GenerationMixin.generate`] does its best effort to infer the attention mask when it is not passed, we recommend to pass it whenever possible for optimal results.

Finally, you can call the [`~generation.GenerationMixin.generate`] method. It returns the generated tokens, which should be converted to text before printing.

```py
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A list of colors: red, blue, green, yellow, black, white, and brown'
```


## Generation with other modalities

Autoregressive generation with other modalities behave mostly as described above for LLMs. As such, let's focus on the differences that you may enconter when generating with other modalities:
* Non-text model inputs rely on the [`AutoProcessor`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor) class for pre-processing;
* If the output of your model's forward pass is not a discrete set (e.g. if they are embeddings), then the logit processing step described above does not apply, but there may be custom model output processing steps between iterations.

And... that's it!

For instance, let's take the image captioning problem.

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image = Image.open(requests.get(url, stream=True).raw)
```

The variable `image` contains a lovely image of two cats.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" alt="Test image"/>
</div>

You can now use the same workflow as above to caption it, replacing the `AutoTokenizer` by the `AutoProcessor` and importing the appropriate model class.

```py
>>> from transformers import AutoProcessor, AutoModelForVision2Seq

>>> model = AutoModelForVision2Seq.from_pretrained(
...     "Salesforce/blip2-opt-2.7b", device_map="auto", load_in_4bit=True
... )
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model_inputs = processor(image, return_tensors="pt").to("cuda")

>>> generated_ids = model.generate(**model_inputs)
>>> processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
'two cats laying on a pink couch\n'
```


## Common pitfalls

Autoregressive generation can be controlled with great precision, as we explain in our [generation strategies guide](generation_strategies). However, before you read our advanced docs, let's go through the most common pitfalls, using an LLM as an example.

```py
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
>>> tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default
>>> model = AutoModelForCausalLM.from_pretrained(
...     "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
... )
```

1. Not controlling the maximum length. If not specified in the [`~generation.GenerationConfig`] file, a `generate` call will return up to `20` tokens (our default value). We highly recommend you manually setting `max_new_tokens` in your generate call -- this flag controls the maximum number of new tokens it can return. Please note that LLMs (more precisely, [decoder-only models](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt)) also return the input prompt as part of the output.


```py
>>> model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")

>>> # By default, the output will contain up to 20 tokens
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'

>>> # Setting `max_new_tokens` allows you to control the maximum length
>>> generated_ids = model.generate(**model_inputs, max_new_tokens=50)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

2. Selecting whether the output is sampled or not. By default, and unless specified in the [`~generation.GenerationConfig`] file, `generate` simply selects the most likely token at each iteration (greedy decoding). Depending on your task, this may be undesirable: creative tasks like being a chatbot or writing an essay benefit from sampling. On the other hand, input-grounded tasks like audio transcription or translation benefit from greedy decoding. You can enable sampling with `do_sample=True`, and we further elaborate on this topic on our [blog post](https://huggingface.co/blog/how-to-generate).

```py
>>> # Set seed or reproducibility -- you don't need this unless you want full reproducibility
>>> from transformers import set_seed
>>> set_seed(0)

>>> model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to("cuda")

>>> # LLM + greedy decoding = repetitive, boring output
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat. I am a cat. I am a cat. I am a cat'

>>> # With sampling, the output becomes more creative!
>>> generated_ids = model.generate(**model_inputs, do_sample=True)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat.\nI just need to be. I am always.\nEvery time'
```

3. Batched LLM inference without left-padding. LLMs are [decoder-only](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt) architectures, which means that they continue your input prompt. If your inputs do not have the same length, they will have to be padded. Since LLMs are not trained to continue from pad tokens, your input needs to be left-padded. Make sure you also don't forget to pass the attention mask to generate!

```py
>>> # The tokenizer initialized above has right-padding active by default: the 1st sequence,
>>> # which is shorter, has padding on the right side. Generation fails.
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
''

>>> # With left-padding, it works as expected!
>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b", padding_side="left")
>>> tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 3, 4, 5, 6,'
```

<!-- TODO: when the prompting guide is ready, mention the importance of setting the right prompt in this section -->

## Further resources

While the core principles of autoregressive generation are straightforward, taking the most out of your generative model can be a challenging endeavour, as there are many moving parts. This section is here to serve as a reference for next steps.

<!-- TODO: complete with new guides -->
### Advanced generate usage
1. [Guide](generation_strategies) on how to control different generation methods, how to set up the generation configuration file, and how to stream the output;
2. API reference on [~generation.GenerationConfig], [~generation.GenerationMixin.generate], and [generate-related classes](internal/generation_utils).

### LLMs
1. [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), which focuses on the quality of the open-source models;
2. [Open LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard), which focuses on LLM throughput.

### Latency and Throughput
1. [Guide](main_classes/quantization) on dynamic quantization, which shows you how to drastically reduce your memory requirements.

### Related libraries
1. [`text-generation-inference`](https://github.com/huggingface/text-generation-inference), a production-ready server for LLMs;
2. [`optimum`](https://github.com/huggingface/optimum), an extension of 🤗 `transformers` that optimizes for specific hardware devices.
