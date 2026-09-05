<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# LiteRT

[LiteRT](https://ai.google.dev/edge/litert) (formerly TensorFlow Lite) is Google's runtime for on-device inference. The model format is still `.tflite`, and language models ship as one `.litertlm` file for the [LiteRT-LM](https://ai.google.dev/edge/litert-lm) runtime.

Export a Transformers model with [litert-torch](https://github.com/google-ai-edge/litert-torch) (formerly `ai-edge-torch`). It lowers the [torch.export](https://docs.pytorch.org/docs/stable/export.html) graph to LiteRT directly, without ONNX or TensorFlow.

```bash
pip install litert-torch
```

<hfoptions id="export">
<hfoption id="CLI (LLM)">

`export_hf` loads a text-generation model from the Hub, quantizes the weights to int8 by default, and writes `model.litertlm`.

```bash
litert-torch export_hf \
    --model="HuggingFaceTB/SmolLM2-135M-Instruct" \
    --output_dir="./smollm2_litertlm"
```

</hfoption>
<hfoption id="Python (any model)">

`litert_torch.convert` traces a model with sample inputs and exports a `.tflite` file. The returned object also runs it, so the export can be checked in place.

```py
import litert_torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).eval()
inputs = tokenizer("Paris is the [MASK] of France.", return_tensors="pt", padding="max_length", max_length=128)

litert_model = litert_torch.convert(model, sample_kwargs=dict(inputs))
litert_model.export("bert.tflite")

outputs = litert_model(**{name: tensor.numpy() for name, tensor in inputs.items()})
mask_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
print(tokenizer.decode(outputs["logits"][0, mask_index].argmax()))  # capital
```

</hfoption>
</hfoptions>

## Transformers integration

1. [`~PreTrainedModel.from_pretrained`] loads the model weights in safetensors format.
2. litert-torch runs [torch.export](https://docs.pytorch.org/docs/stable/export.html) and lowers the graph to LiteRT operators. `export_hf` adds the KV cache, prefill and decode signatures, and int8 quantization.
3. [`AutoTokenizer`] loads the tokenizer. `export_hf` packs it and the chat template into the `.litertlm` file.
4. At runtime, `.tflite` runs on LiteRT and `.litertlm` on LiteRT-LM, from Kotlin, Swift, C++, or Python (`ai-edge-litert` and `litert-lm-api`; the older `tflite-runtime` wheels stop at Python 3.11).

> [!NOTE]
> Transformers v4 documented `optimum-cli export tflite`, which converted through TensorFlow. It was removed with TensorFlow support in v5 ([#40760](https://github.com/huggingface/transformers/pull/40760)) and is not part of Optimum 2.x.

## Resources

- [LiteRT](https://ai.google.dev/edge/litert) and [LiteRT-LM](https://ai.google.dev/edge/litert-lm) docs
- [Convert PyTorch models](https://ai.google.dev/edge/litert/conversion/pytorch/overview) and [GenAI models](https://ai.google.dev/edge/litert/conversion/pytorch/genai) guides
