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

# Choose a method

If your checkpoint is already quantized, the checkpoint decides the method. Load it as shown in [Load a pre-quantized checkpoint](./overview#load-a-pre-quantized-checkpoint). Otherwise, pick a method based on your hardware and what you need to do with the model after loading it.

## Start from your goal

Each row recommends a first method to try. Check the [comparison table](#compare-methods) before you commit to one.

| Goal | Start with | Why |
|---|---|---|
| Fit a model in less memory with the least setup | [bitsandbytes](./bitsandbytes) | Quantizes to 4 or 8 bits while loading without calibration, runs on CUDA, XPU, HPU, and CPU, and supports saving and fine-tuning. |
| Fine-tune with QLoRA | [bitsandbytes](./bitsandbytes#qlora) | Supports 4-bit NF4 with PEFT. [HQQ](./hqq), [EETQ](./eetq), [GPTQ](./gptq), [SINQ](./sinq), and [Quanto](./quanto) also support fine-tuning. |
| Speed up inference with `torch.compile` | [torchao](./torchao) | `generate` compiles torchao models automatically with a static cache, and torchao has int4, int8, and FP8 schemes for CUDA, XPU, and CPU. |
| Get the best 4-bit accuracy | [GPTQ](./gptq) | Calibrates on a dataset while loading. To use [AWQ](./awq) or [AutoRound](./auto_round) instead, quantize with their libraries and load the checkpoint. |
| Run FP8 on Ada, Hopper, or Intel XPU | [Fine-grained FP8](./finegrained_fp8) | Needs CUDA compute capability 8.9 or higher, or an XPU. [FBGEMM FP8](./fbgemm_fp8) needs compute capability 9.0 or higher, or an XPU. |
| Run FP4 on Blackwell | [NVFP4](./nvfp4) | Quantizes to NVIDIA's 4-bit float format while loading. [MXFP4](./mxfp4) runs on compute capability 7.5 or higher, XPU, and CPU. |
| Run on Apple silicon | [Metal](./metal) | Quantizes to 2, 4, or 8 bits on MPS while loading. [GGUF](./gguf) files also load on MPS. |
| Load checkpoints made with llm-compressor | [compressed-tensors](./compressed_tensors) | Loads FP8, int4, and int8 checkpoints in the compressed-tensors format. |
| Compress below 4 bits | [HQQ](./hqq) or [SINQ](./sinq) | Quantize to 2 or 3 bits while loading without calibration. [AQLM](./aqlm), [VPTQ](./vptq), and [SpQR](./spqr) checkpoints go lower but must be quantized with their own libraries. |

## Compare methods

The table lists every method Transformers supports, in alphabetical order.

- Workflow says whether Transformers can quantize the model while loading it, or can only load a checkpoint quantized somewhere else.
- `quant_method` is the value stored in a checkpoint's `quantization_config`.
- Save means [`~PreTrainedModel.save_pretrained`] writes the quantized weights.
- Fine-tune means [`Trainer`] accepts the quantized model for training, for example with PEFT adapters.
- Auto-compile means `generate` compiles the model automatically when it uses a static cache. A ❌ doesn't always mean `torch.compile` fails. Check the method's page.

| Method | `quant_method` | Workflow | Bits | Hardware | Save | Fine-tune | Auto-compile |
|---|---|---|---|---|---|---|---|
| [AQLM](./aqlm) | `aqlm` | Pre-quantized only | 1–2 | CUDA, CPU | ✅ | ✅ with `aqlm>=1.0.2` | ❌ |
| [AutoRound](./auto_round) | `auto-round` | Pre-quantized only | 2–8 | CUDA, XPU, CPU | ✅ | ❌ | ❌ |
| [AWQ](./awq) | `awq` | Pre-quantized only | 4 | CUDA, ROCm, XPU, CPU | ✅ except ExLlama backends | ✅ with `gptqmodel>=5.0.0` | ❌ |
| [BitNet](./bitnet) | `bitnet` | Pre-quantized only | 1.58 | CUDA | ✅ | Only `autobitlinear` in `online` mode | ❌ |
| [bitsandbytes](./bitsandbytes) | `bitsandbytes` | While loading | 4, 8 | CUDA, XPU, HPU, CPU | ✅ | ✅ | ❌ |
| [compressed-tensors](./compressed_tensors) | `compressed-tensors` | Pre-quantized only | Set by checkpoint | CUDA, XPU, MPS, CPU | ✅ | ✅ except with FP8 kernels | ✅ |
| [EETQ](./eetq) | `eetq` | While loading | 8 | CUDA | ✅ | ✅ | ❌ |
| [FBGEMM FP8](./fbgemm_fp8) | `fbgemm_fp8` | While loading | 8 | CUDA (compute capability 9.0+), XPU | ✅ | ❌ | ❌ |
| [Fine-grained FP8](./finegrained_fp8) | `fp8`, `mxfp8` | While loading | 8 | CUDA (compute capability 8.9+), XPU | ✅ | ❌ | ✅ |
| [Four Over Six](./fouroversix) | `fouroversix` | While loading | 4 | CUDA (fastest on Blackwell), CPU | ✅ | Only with `keep_master_weights=True` | ❌ |
| [FP-Quant](./fp_quant) | `fp_quant` | While loading | 4 | CUDA, XPU | ✅ | Only with `store_master_weights=True` | ❌ |
| [GGUF](./gguf) | `gguf` | Pre-quantized only | 1–8 | CUDA, MPS, XPU, CPU | ❌ | ❌ | ✅ |
| [GPTQ](./gptq) | `gptq` | While loading, with calibration | 2, 3, 4, 8 | CUDA, ROCm, XPU, CPU | ✅ | ✅ | ❌ |
| [HIGGS](./higgs) | `higgs` | While loading | 2, 3, 4 | CUDA | ✅ | ❌ | ❌ |
| [HQQ](./hqq) | `hqq` | While loading | 1, 2, 3, 4, 8 | CUDA, XPU | ✅ | ✅ | ❌ |
| [Metal](./metal) | `metal` | While loading | 2, 4, 8 | MPS | ✅ | ❌ | ❌ |
| [MXFP4](./mxfp4) | `mxfp4` | While loading | 4 | CUDA (compute capability 7.5+), XPU, CPU | ✅ | ❌ | ❌ |
| [NVFP4](./nvfp4) | `nvfp4` | While loading | 4 | One Blackwell GPU | ❌ | ❌ | ✅ |
| [Quanto](./quanto) | `quanto` | While loading | 2, 4, 8 | CUDA, MPS, XPU, CPU | ❌ | ✅ | ❌ |
| [Quark](./quark) | `quark` | Pre-quantized only | Set by checkpoint | CUDA, ROCm, CPU | ❌ | ❌ | ❌ |
| [SINQ](./sinq) | `sinq` | While loading | 2, 3, 4, 6, 8 | One CUDA GPU, CPU | ✅ | ✅ | ❌ |
| [SpQR](./spqr) | `spqr` | Pre-quantized only | 3 | CUDA | ✅ | ❌ | ❌ |
| [torchao](./torchao) | `torchao` | While loading | 4, 8 | CUDA, XPU, CPU | ✅ | Only 8-bit schemes | ✅ |
| [VPTQ](./vptq) | `vptq` | Pre-quantized only | 1–8 | CUDA | ✅ | ❌ | ❌ |

## Check memory with the quantization picker

The [quantization picker](https://huggingface.co/spaces/stevhliu/quantization-picker) estimates whether a model fits on your hardware at each precision, including the KV cache. Pick a model, your hardware, and what you need the method to do, and the **Compare** view lists the methods that match.

<iframe
	src="https://stevhliu-quantization-picker.hf.space"
	frameborder="0"
	width="100%"
	height="420"
></iframe>

## Benchmarks

The benchmarks below compare several methods on Llama 3.1 8B and 70B. They report accuracy, throughput in tokens per second, peak memory in GB, and quantization time.

Llama 3.1 70B in bf16 ran on 2 NVIDIA A100 80GB GPUs. FP8 methods ran on 1 NVIDIA H100 80GB GPU, and all other methods ran on 1 NVIDIA A100 80GB GPU. Throughput was measured at batch size 1, generating 64 tokens. Results include `torch.compile` and Marlin kernels where a method supports them.

<iframe
  src="https://huggingface.co/datasets/derekl35/quantization-benchmarks/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
  title="benchmarking results dataset"
></iframe>

The results group into three bit ranges.

| Bits | Methods benchmarked | Memory vs. bf16 | Accuracy |
|---|---|---|---|
| 8 | bitsandbytes int8, HQQ, Quanto, torchao, FP8 | About 2x smaller | Very close to bf16 |
| 4 | AWQ, GPTQ, HQQ, bitsandbytes NF4, SINQ | About 4x smaller | High. AWQ and GPTQ often lead but need calibration. |
| Below 4 | VPTQ, AQLM, 2-bit GPTQ | More than 4x smaller | Noticeable drop, especially at 2 bits. AQLM and VPTQ take a long time to quantize. |

> [!TIP]
> Benchmark accuracy and speed on your own task and hardware before you pick a method for production.
