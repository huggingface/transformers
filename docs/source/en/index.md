<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Transformers

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformers is the model-definition framework for machine learning models in text, computer vision, audio, video, and multimodal tasks. It defines each model once, in PyTorch, and you use that definition for both inference and training.

The rest of the ecosystem builds on the model definition instead of writing its own. Training frameworks, inference engines, and local runtimes all use the model definition from Transformers, so a model supported in Transformers is compatible with most of these tools.

<div class="not-prose relative flex items-center mb-28">
  <div class="peer group pr-4">
    <a class="flex h-10 items-center" href="./community_integrations/vllm">
      <span class="relative block h-10 w-10">
        <img class="!m-0 !border-0 !shadow-none !rounded-none object-contain h-full w-full grayscale opacity-50" src="https://cdn.jsdelivr.net/gh/homarr-labs/dashboard-icons/svg/vllm.svg" alt="vLLM"/>
        <img class="!m-0 !border-0 !shadow-none !rounded-none object-contain absolute inset-0 h-full w-full opacity-0 group-hover:opacity-100" src="https://cdn.jsdelivr.net/gh/homarr-labs/dashboard-icons/svg/vllm.svg" alt=""/>
      </span>
    </a>
    <div class="invisible group-hover:visible absolute left-0 top-full w-full pt-3">
      <div class="text-gray-500 dark:text-gray-400">vLLM</div>
      Serve LLMs at scale. vLLM batches requests continuously and keeps KV cache memory compact with PagedAttention.
    </div>
  </div>
  <div class="peer group pr-4">
    <a class="flex h-10 items-center" href="./community_integrations/sglang">
      <span class="relative block h-6 w-20">
        <img class="!m-0 !border-0 !shadow-none !rounded-none object-contain h-full w-full grayscale opacity-50" src="https://www.sglang.io/images/logo.png" alt="SGLang"/>
        <img class="!m-0 !border-0 !shadow-none !rounded-none object-contain absolute inset-0 h-full w-full opacity-0 group-hover:opacity-100" src="https://www.sglang.io/images/logo.png" alt=""/>
      </span>
    </a>
    <div class="invisible group-hover:visible absolute left-0 top-full w-full pt-3">
      <div class="text-gray-500 dark:text-gray-400">SGLang</div>
      Serve LLMs with low latency. SGLang is a high-throughput inference engine with a frontend language for agentic workflows.
    </div>
  </div>
  <div class="peer group pr-4">
    <a class="flex h-10 items-center" href="./community_integrations/llama_cpp">
      <span class="relative block h-7 w-7">
        <img class="!m-0 !border-0 !shadow-none !rounded-none object-contain h-full w-full grayscale opacity-50" src="https://cdn.jsdelivr.net/gh/selfhst/icons/svg/llama-cpp.svg" alt="llama.cpp"/>
        <img class="!m-0 !border-0 !shadow-none !rounded-none object-contain absolute inset-0 h-full w-full opacity-0 group-hover:opacity-100" src="https://cdn.jsdelivr.net/gh/selfhst/icons/svg/llama-cpp.svg" alt=""/>
      </span>
    </a>
    <div class="invisible group-hover:visible absolute left-0 top-full w-full pt-3">
      <div class="text-gray-500 dark:text-gray-400">llama.cpp</div>
      Run LLMs locally without Python. llama.cpp is a C/C++ inference engine that runs quantized models in the GGUF format.
    </div>
  </div>
  <div class="order-first pr-4 peer-hover:hidden">
    <a class="flex h-10 items-center" href="./community_integrations/unsloth"><img class="!m-0 !border-0 !shadow-none !rounded-none object-contain h-8 w-8" src="https://cdn.jsdelivr.net/gh/selfhst/icons/webp/unsloth.webp" alt="Unsloth"/></a>
  </div>
  <div class="order-first pr-4 hidden peer-hover:block">
    <a class="flex h-10 items-center" href="./community_integrations/unsloth"><img class="!m-0 !border-0 !shadow-none !rounded-none object-contain h-8 w-8 grayscale opacity-50" src="https://cdn.jsdelivr.net/gh/selfhst/icons/webp/unsloth.webp" alt=""/></a>
  </div>
  <div class="peer-hover:hidden absolute left-0 top-full w-full pt-3">
    <div class="text-gray-500 dark:text-gray-400">Unsloth</div>
    Fine-tune with less memory. Unsloth trains LLMs in 4-bit, 8-bit, and 16-bit precision with custom kernels.
  </div>
</div>

Transformers implements each model to match the results of its original release. A tool that builds on Transformers gets each new model without implementing it again.

## Try a model

Install Transformers with PyTorch. See the [Installation](./installation) guide for other setups.

```bash
uv pip install "transformers[torch]"
```

Create a [`Pipeline`] with a chat model from the [Hugging Face Hub](https://huggingface.co/models?library=transformers&sort=trending) and pass it a message.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen3-0.6B")
messages = [
    {"role": "user", "content": "What's the secret to baking a really good cake? Answer in one sentence."},
]
output = pipeline(messages, tokenizer_encode_kwargs={"enable_thinking": False})
print(output[0]["generated_text"][-1]["content"])
# The secret to baking a really good cake is combining the perfect balance of ingredients, technique, and a warm, loving atmosphere.
```

The [Quickstart](./quicktour) covers loading models, running inference on other tasks, and training.

## Set up your coding agent

Copy the prompt below into your coding agent. It has the agent check your installed version, read the docs for that version, and follow the best and latest practices from Transformers v5.

> [!AgentPrompt]
> I'm building with Hugging Face Transformers. Before you write any Transformers code:
> 
> 1. Check the installed version with `python -c "import transformers; print(transformers.__version__)"`. If Transformers isn't installed, install it with `pip install "transformers[torch]"`.
> 2. Read the docs index for that version at https://huggingface.co/docs/transformers/v<version>/llms.txt, for example v5.22.2. For a .dev version, use https://huggingface.co/docs/transformers/main/llms.txt. Add .md to any docs page URL to read it as Markdown. Look up each class and argument there instead of relying on memory.
> 3. If the version is 5.0 or later, read https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md Don't use APIs from older versions.
> 4. Read the model card of each model you use on the Hugging Face Hub. Many models need specific arguments, like a chat template option or a minimum Transformers version.

## Run inference

Load a pretrained model and run it on text, images, audio, video, or any combination of modalities.

<div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-4">
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./pipeline_tutorial">
    <div class="font-semibold">Pipeline</div>
    <p class="text-gray-500 dark:text-gray-400">Preprocess the input, run the model, and postprocess the output for a task with one class.</p>
  </a>
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./llm_tutorial">
    <div class="font-semibold">Text generation</div>
    <p class="text-gray-500 dark:text-gray-400">Generate text with LLMs and VLMs, and control the decoding strategy and streaming.</p>
  </a>
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./conversations">
    <div class="font-semibold">Chat with models</div>
    <p class="text-gray-500 dark:text-gray-400">Start a chat in your terminal with <code>transformers chat</code>, or pass a chat history to a pipeline.</p>
  </a>
</div>

## Serve models

Serve a model to other applications over HTTP. `transformers serve` is for evaluation, experimentation, and moderate load. For large production deployments, use an inference engine with a Transformers model as the backend.

<div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-4">
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./serve-cli/serving">
    <div class="font-semibold">Serve CLI</div>
    <p class="text-gray-500 dark:text-gray-400">Start a local server compatible with the OpenAI SDK with <code>transformers serve</code>.</p>
  </a>
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./continuous_batching">
    <div class="font-semibold">Continuous batching</div>
    <p class="text-gray-500 dark:text-gray-400">Raise throughput by adding new requests to the batch as other requests finish.</p>
  </a>
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./community_integrations/transformers_as_backend">
    <div class="font-semibold">Inference engines</div>
    <p class="text-gray-500 dark:text-gray-400">Use a Transformers model as the backend for vLLM or SGLang.</p>
  </a>
</div>

## Train models

Fine-tune a pretrained model on your data, on one GPU or many.

<div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-4">
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./training">
    <div class="font-semibold">Trainer</div>
    <p class="text-gray-500 dark:text-gray-400">Run the training loop with support for mixed precision and <code>torch.compile</code>.</p>
  </a>
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./peft">
    <div class="font-semibold">PEFT</div>
    <p class="text-gray-500 dark:text-gray-400">Train a small adapter instead of all of a model's weights.</p>
  </a>
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./fsdp">
    <div class="font-semibold">FSDP</div>
    <p class="text-gray-500 dark:text-gray-400">Shard a model's parameters across GPUs to train models that don't fit on one.</p>
  </a>
</div>

## Use less memory and run faster

Fit a model on your hardware and speed it up.

<div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-4">
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./quantization/overview">
    <div class="font-semibold">Quantization</div>
    <p class="text-gray-500 dark:text-gray-400">Store weights in fewer bits so a model needs less memory.</p>
  </a>
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./kernel_doc/overview">
    <div class="font-semibold">Kernels</div>
    <p class="text-gray-500 dark:text-gray-400">Load precompiled operations for your hardware from the Hub.</p>
  </a>
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./attention_interface">
    <div class="font-semibold">Attention backends</div>
    <p class="text-gray-500 dark:text-gray-400">Switch between attention implementations without changing the model.</p>
  </a>
</div>

## Add or change a model

Add a model to Transformers or edit an existing one.

<div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-4">
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./modular_transformers">
    <div class="font-semibold">Modular Transformers</div>
    <p class="text-gray-500 dark:text-gray-400">Add a model by inheriting from existing models. A converter generates the standalone file.</p>
  </a>
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./modeling_rules">
    <div class="font-semibold">Model structure rules</div>
    <p class="text-gray-500 dark:text-gray-400">Follow the conventions a checker enforces on every modeling, modular, and configuration file.</p>
  </a>
  <a class="!no-underline border dark:border-gray-700 p-4 rounded-lg hover:shadow-lg" href="./testing">
    <div class="font-semibold">Writing model tests</div>
    <p class="text-gray-500 dark:text-gray-400">Write a small amount of model-specific test code, and the test mixins generate more than 100 tests from it.</p>
  </a>
</div>
