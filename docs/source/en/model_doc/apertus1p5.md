<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-07-14.*


# Apertus 1.5

> [!WARNING]
> The vision tokenizer (`Apertus1p5VQVAE`) must run in `float32`: code assignment is an argmax over codebook
> logits, and half precision flips a significant fraction of codes (~8% in bf16 at the 131k codebook). It is
> kept in `float32` automatically when the model is loaded with `dtype=torch.float16`/`bfloat16`
> (`_keep_in_fp32_modules_strict`).

> [!WARNING]
> `Apertus1p5VQVAE` is an inference-only vision-tokenizer port. It does not support training the vision tokenizer:
> the port implements only inference-time codebook scoring and hard `argmax` indices, and omits IBQ's
> differentiable index-backpropagation path, tokenizer training losses, and decoder. The hard indices also stop
> gradients from the language-model loss at the tokenizer boundary. Calling `.train()` does not restore the
> omitted training implementation. This limitation applies to the vision tokenizer; the Apertus language model
> retains the standard differentiable Transformers forward and loss APIs.

`Apertus1p5VQVAE.encode` intentionally does not force `torch.no_grad`. Public Transformers model methods normally
respect the caller's ambient gradient mode, so applications retain explicit control over it. This convention does
not imply training support: use `torch.no_grad()` or `torch.inference_mode()` for standalone tokenization to avoid
retaining unnecessary encoder activations.

## Overview

The Apertus 1.5 model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

<INSERT PAPER ABSTRACT HERE>

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).

## Usage examples

<INSERT SOME NICE EXAMPLES HERE>

## Apertus1p5Config

[[autodoc]] Apertus1p5Config

## Apertus1p5TextConfig

[[autodoc]] Apertus1p5TextConfig

## Apertus1p5VQVAEConfig

[[autodoc]] Apertus1p5VQVAEConfig

## Apertus1p5ForConditionalGeneration

[[autodoc]] Apertus1p5ForConditionalGeneration

## Apertus1p5ForCausalLM

[[autodoc]] Apertus1p5ForCausalLM

## Apertus1p5TextModel

[[autodoc]] Apertus1p5TextModel
    - forward

## Apertus1p5VQVAE

[[autodoc]] Apertus1p5VQVAE
    - encode

## Apertus1p5Model

[[autodoc]] Apertus1p5Model
    - forward

## Apertus1p5ImageProcessor

[[autodoc]] Apertus1p5ImageProcessor

## Apertus1p5Processor

[[autodoc]] Apertus1p5Processor
