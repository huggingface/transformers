<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ExecuTorch

[ExecuTorch](https://docs.pytorch.org/executorch/stable/index.html) is a lightweight runtime for model inference on edge devices. It exports a PyTorch model into a portable, ahead-of-time format. A small C++ runtime plans memory and dispatches operations to hardware-specific backends. Execution and memory behavior is known before the model runs on device, so inference overhead is low.

Export a Transformers model with the [optimum-executorch](https://huggingface.co/docs/optimum-executorch/en/index) library.

<hfoptions id="export">
<hfoption id="CLI">

```bash
optimum-cli export executorch \
    --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --output_dir="./smollm2_exported"
```

</hfoption>
<hfoption id="Python">

```py
from transformers import AutoTokenizer
from optimum.executorch import ExecuTorchModelForCausalLM

model = ExecuTorchModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    recipe="xnnpack",
)
model.save_pretrained("./smollm2_exported")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
```

</hfoption>
</hfoptions>

## Transformers integration

The export process uses several Transformers components.

1. [`~PreTrainedModel.from_pretrained`] loads the model weights in safetensors format.
2. Optimum applies graph optimizations and runs [torch.export](https://docs.pytorch.org/docs/stable/export.html) to create a `model.pte` file targeting your hardware backend.
3. [`AutoTokenizer`] or [`AutoProcessor`] loads the tokenizer or processor files and runs during inference.
4. At runtime, a C++ runner class executes the `.pte` file on the ExecuTorch runtime.

```c++
#include <executorch/extension/llm/runner/text_llm_runner.h>

using namespace executorch::extension::llm;

int main() {
  // Load tokenizer and create runner
  auto tokenizer = load_tokenizer("path/to/tokenizer.json", nullptr, std::nullopt, 0, 0);
  auto runner = create_text_llm_runner("path/to/model.pte", std::move(tokenizer));

  // Load the model
  runner->load();

  // Configure generation
  GenerationConfig config;
  config.max_new_tokens = 100;
  config.temperature = 0.8f;

  // Generate text with streaming output
  runner->generate("The capital of France is", config,
    [](const std::string& token) { std::cout << token << std::flush; },
    nullptr);

  return 0;
}
```

## Resources

- [ExecuTorch](https://docs.pytorch.org/executorch/stable/index.html) docs
- [torch.export](https://docs.pytorch.org/docs/stable/export.html) docs
- [Exporting to production](../serialization#executorch) guide
