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

# Apple Silicon

Apple Silicon (M-series) chips run PyTorch through the [Metal Performance Shaders (MPS)](https://docs.pytorch.org/docs/stable/notes/mps.html) backend, with the device string `mps`. The CPU and GPU share one pool of unified memory.

MPS works with the standard `torch` install. Current PyTorch releases for Apple Silicon require macOS 14 or later. Check that PyTorch was built with MPS and can see the GPU.

```py
import torch

print(torch.backends.mps.is_available())
```

It prints `True` when MPS is ready to use.

## Choose an attention backend

Set the attention backend with `attn_implementation` when you load a model, and it applies to both inference and training. Models use [SDPA](./attention_interface) by default. With PyTorch 2.13 or later, SDPA on MPS handles grouped-query attention directly, without repeating the key and value heads first.

Use SDPA for training.

- [kernels-community/metal-flash-sdpa](https://huggingface.co/kernels-community/metal-flash-sdpa) only supports variable-length inputs, such as packed sequences or a batch whose attention mask contains padding. A batch without padding, such as a single prompt, raises an error. It doesn't compute gradients for the query, key, and value projections. [gpt-oss](./model_doc/gpt_oss) switches to it automatically when you request flash attention on MPS.
- [ggml-org/ggml-attn](https://huggingface.co/kernels/ggml-org/ggml-attn) is the default for `transformers serve` and for [GGUF](./quantization/gguf#attention) models when `kernels` is installed.

Install the `kernels` extra to use either kernel. It installs a version that Transformers supports.

```bash
pip install "transformers[kernels]"
```

## Run inference

Pass `device_map="mps"` to [`~PreTrainedModel.from_pretrained`]. [`pipeline`] picks MPS automatically when you don't pass `device`.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

inputs = tokenizer("Apple Silicon is", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

With a `device_map` that targets MPS, Transformers loads safetensors weights directly onto the MPS device.

### Serve a model

[transformers serve](./serve-cli/serving) starts an OpenAI-compatible server. Its `--device` option defaults to `auto`, so it loads models on MPS without extra flags. Install the serving dependencies and start the server.

```bash
pip install "transformers[serving]"
transformers serve
```

The server loads a model the first time a request names it. Send a chat request to `http://localhost:8000`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What is Apple Silicon? Answer in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

```json
{"id":"b48e84c1-b3c6-484d-b781-4c9ce8022dff","choices":[{"finish_reason":"stop","index":0,"message":{"content":"Apple Silicon is a type of processor technology developed and manufactured by Apple Inc., designed to offer faster performance and lower power consumption compared to traditional silicon-based processors. This technology allows for the creation of more efficient and energy-efficient devices, such as high-performance computers, smartphones, and other Apple products.","role":"assistant"}}],"created":1791843839,"model":"Qwen/Qwen2.5-0.5B-Instruct@main","object":"chat.completion","usage":{"completion_tokens":59,"prompt_tokens":39,"total_tokens":98}}
```

When `kernels` is installed, `transformers serve` defaults to [ggml-org/ggml-attn](https://huggingface.co/kernels/ggml-org/ggml-attn) on MPS and logs a warning at startup. Pass `--attn-implementation sdpa` to use SDPA instead.

## Quantize models

The model weights share unified memory with macOS and everything else running on the Mac, so quantizing them lets a larger model fit for inference or fine-tuning. Metal quantizes a model when you load it, while GGUF runs checkpoints that are already quantized. Both are inference-only.

- [Metal](./quantization/metal) quantizes to 2, 4, or 8 bits with Metal kernels from the Hub.
- [GGUF](./quantization/gguf) keeps weights packed on MPS when the `ggml-org/ggml-quantization` kernel is available.
- [bitsandbytes](./quantization/bitsandbytes) quantizes to 4 bits with native MPS kernels.

To fine-tune a model that doesn't fit in memory, load it in 4-bit with bitsandbytes and train LoRA adapters with [PEFT](./peft).

## Train with Trainer

[`TrainingArguments`] detects MPS and sets the device to `mps`, so [`Trainer`] trains on it without any device settings. Both bf16 and fp16 mixed precision work. Load the model with the default SDPA attention, because [metal-flash-sdpa](#choose-an-attention-backend) doesn’t compute gradients for the attention projections.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
)
print(training_args.device)
```

It prints `mps` when `Trainer` will train on the GPU.

On macOS, Python starts worker processes with `spawn` by default, which re-imports your script and requires a picklable dataset and `collate_fn`. When `dataloader_num_workers` is greater than 1, [`TrainingArguments`] sets `dataloader_multiprocessing_context="fork"` on MPS, so workers start from a copy of the training process instead. Set it to `"spawn"` if a data source isn't fork-safe.

### Limit graph cache growth

MPS compiles a Metal kernel for each tensor shape it sees and keeps them in a graph cache that is never evicted. Training with variable-length inputs adds a kernel for every new shape and can run out of unified memory.

Reduce the number of shapes by padding each batch to a multiple of a fixed length with [`DataCollatorWithPadding`]'s `pad_to_multiple_of` argument.

```py
from transformers import AutoTokenizer, DataCollatorWithPadding

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64)
```

Pass it to `Trainer` with `data_collator=data_collator`.

## Limitations

- `device_map="auto"` can't offload layers to the CPU, because MPS and the CPU share the same memory. Overflow goes to disk.
- [compressed-tensors](./quantization/compressed_tensors) FP8 checkpoints load on MPS, but their weights are decompressed by the first forward pass, so they don't reduce memory during inference.
- Tensor parallelism isn't supported because a Mac only has one device.
- Some PyTorch operations don't have an MPS implementation yet. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to run them on the CPU instead of raising an error.

## Next steps

- Read the PyTorch [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html) notes for supported operations and known issues.
- Read the [Attention backends](./attention_interface) guide to learn how `attn_implementation` selects an attention function.
- Check the Metal column in the [quantization overview](./quantization/overview) to find every quantization method that runs on MPS.
