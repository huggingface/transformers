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

# Chat basics

Chat models are conversational models which you communicate with by sending and receiving a series of messages, like an online chat. Most new language models from mid-2023 onwards are chat models, and models which are not trained for chat are usually referred to as "base" models, while models trained for chat are sometimes called "instruct" or "instruction-tuned". There are many chat models available to choose from; larger and newer models tend to be more capable, though there are plenty of exceptions to this rule!

If you're just looking for a general chat model, try leaderboards like [OpenLLM](https://hf.co/spaces/HuggingFaceH4/open_llm_leaderboard) and [LMSys Chatbot Arena](https://chat.lmsys.org/?leaderboard) to help you identify the top performers. Be careful, though! Models that are specialized in certain domains (medical/legal text, non-English languages, etc.) can often outperform larger general purpose models, so the top leaderboard models may not be the best at your particular task.

This guide shows you how to quickly load chat models in Transformers from the command line, how to build and format a conversation, and how to chat using the [`TextGenerationPipeline`].

## chat CLI

After you've [installed Transformers](./installation), you can chat with a model directly from the command line. The command below launches an interactive session with a model, with a few base commands listed at the start of the session.

```bash
transformers chat Qwen/Qwen2.5-0.5B-Instruct
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers-chat-cli.png"/>
</div>

You can launch the CLI with arbitrary `generate` flags, with the format `arg_1=value_1 arg_2=value_2 ...`

```bash
transformers chat Qwen/Qwen2.5-0.5B-Instruct do_sample=False max_new_tokens=10
```

For a full list of options, run the command below.

```bash
transformers chat -h
```

The chat is implemented on top of the [AutoClass](./model_doc/auto), using tooling from [text generation](./llm_tutorial) and [chat](./chat_templating). It uses the `transformers serve` CLI under the hood ([docs](./serving.md#serve-cli)).


## TextGenerationPipeline

[`TextGenerationPipeline`] is a high-level text generation class with a "chat mode". Chat mode is enabled when a conversational model is detected and the chat prompt is [properly formatted](./llm_tutorial#wrong-prompt-format).

Chat models accept a chat history as input, which is a list of messages. Each message is a dictionary with `role` and `content` keys.
To start the chat, you can just have a single `user` message. You can also optionally include a `system` message to give the model directions on how to behave.

```py
chat = [
    {"role": "system", "content": "You are a helpful science assistant."},
    {"role": "user", "content": "Hey, can you explain gravity to me?"}
]
```

Create the [`TextGenerationPipeline`] and pass `chat` to it. For large models, setting [device_map="auto"](./models#big-model-inference) helps load the model quicker and automatically places it on the fastest device available.

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype="auto", device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

If this works successfully, you should see a response from the model! If you want to continue the conversation,
you need to update the chat history with the model's response. You can do this either by appending the text
to `chat` (use the `assistant` role), or by reading `response[0]["generated_text"]`, which contains
the full chat history, including the most recent response.

Once you have the model's response, you can continue the conversation by appending a new `user` message to the chat history, like so:

```py
chat = response[0]["generated_text"]
chat.append(
    {"role": "user", "content": "Woah! But can it be reconciled with quantum mechanics?"}
)
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

By repeating this process, you can continue the conversation as long as you like, at least until the model runs out of context window
or you run out of memory.

## Performance and memory usage

Transformers load models in full `float32` precision by default, and for a 8B model, this requires ~32GB of memory! You can reduce memory usage using the `torch_dtype="auto"` argument, which will generally use `bfloat16` for models that were trained with it. To go even lower, you can quantize the model to 8-bit or 4-bit with [bitsandbytes](https://hf.co/docs/bitsandbytes/index).

> [!TIP]
> Refer to the [Quantization](./quantization/overview) docs for more information about the different quantization backends available.

To load in 8-bit precision, create a [`BitsAndBytesConfig`] with your desired quantization settings and pass it to the pipelines `model_kwargs` parameter. The example below quantizes a model to 8-bits.

```py
from transformers import pipeline, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", model_kwargs={"quantization_config": quantization_config})
```

In general, model size and performance are directly correlated. Larger models are slower in addition to requiring more memory because each active parameter must be read from memory for every generated token. 
This turns out to be the bottleneck for generating text from an LLM, which means that the main options for improving generation speed are to either quantize a model or use hardware with higher memory bandwidth. Adding
more compute power has surprisingly little effect!

You can also try techniques like [speculative decoding](./generation_strategies#speculative-decoding), where a smaller model generates candidate tokens that are verified by the larger model. If the candidate tokens are correct, the larger model can generate more than one token at a time. This significantly alleviates the bandwidth bottleneck and improves generation speed.

> [!TIP]
> MoE models such as [Mixtral](./model_doc/mixtral), [Qwen2MoE](./model_doc/qwen2_moe), and [GPT-OSS](./model_doc/gpt-oss) have lots of parameters, but only "activate" a small fraction of them to generate each token. As a result, MoE models generally have much lower memory bandwidth requirements and can be faster than a regular LLM of the same size. However, techniques like speculative decoding are ineffective with MoE models because more parameters become activated with each new speculated token.
