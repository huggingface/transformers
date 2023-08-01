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

Autoregressive generation is the inference-time procedure of iterativelly calling a model with its own generated
outputs, given a few initial inputs. This procedure, well explained in
[our blog post](https://huggingface.co/blog/how-to-generate), is used with several tasks in different modalities,
including:
* [Causal language modeling](tasks/masked_language_modeling)
* [Translation](tasks/translation)
* [Summarization](tasks/summarization)
* [Automatic speech recognition](tasks/asr)
* [Image captioning](tasks/image_captioning)
* [Text to speech](tasks/text-to-speech)

Despite the glaring input differences, autoregressive generation in 🤗 `transformers` shares the same core principles
and interface across use cases.

This guide will show you how to:

* Use your model with autoregressive generation
* Avoid common pitfalls
* Take the most of your generative model

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers bitsandbytes>=0.39.0 -q
```


## Generation with an LLM

Let's start with the original and most popular use-case of autoregressive generation with transformers: language
models.

<!-- [GIF 1 -- FWD PASS] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_1_1080p.mov"
    ></video>
</figure>


<!-- [GIF 2 -- TEXT GENERATION] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_2_1080p.mov"
    ></video>
</figure>



## Generation with non-text input

## Common pitfalls

## Further references
