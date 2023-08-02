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

Despite the glaring input differences, autoregressive generation in 🤗 `transformers` shares the same core principles and interface across use cases.

This guide will show you how to:

* Use your model with autoregressive generation
* Avoid common pitfalls
* Take the most of your generative model

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers bitsandbytes>=0.39.0 -q
```


## Generation with LLMs

Let's start with the original and most popular use case of autoregressive generation with transformers: language models. A language model trained on causal language modeling will take a sequence of text tokens as input, and returns the probability distribution for the next token. Here's how your LLM's forward pass looks like:

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
>>> model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt")
```

The `model_inputs` variable holds the tokenized text input, as well as the attention mask. While [`~generation.GenerationMixin.generate`] does its best effort to infer the attention mask when it is not passed, we recommend to pass it whenever possible for optimal results.

Finally, you can call the [`~generation.GenerationMixin.generate`] method. It returns the generated tokens, which should be converted to text before printing.

```py
>>> generated_output = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_output, skip_special_tokens=True)
['A list of colors: red, blue, green, yellow, black, white, and brown']
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

You can now use the same workflow as above, replacing the `AutoTokenizer` by the `AutoProcessor` and importing the appropriate model class.

```py
from transformers import AutoProcessor,
```




## Common pitfalls

## Further resources

While the core principles of autoregressive generation are simple, taking the most out of your generative model can be a challenging endeavour. This section is here to guide you on next steps.

### Advanced generate usage
aaa

### LLMs
aaa
### Performance
aaa

### API reference
aaa

### Related libraries
aaa
