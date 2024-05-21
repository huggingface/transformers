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

# Quantization

Quantization techniques focus on representing data with less information while also trying to not lose too much accuracy. This often means converting a data type to represent the same information with fewer bits. For example, if your model weights are stored as 32-bit floating points and they're quantized to 16-bit floating points, this halves the model size which makes it easier to store and reduces memory-usage. Lower precision can also speedup inference because it takes less time to perform calculations with fewer bits.

As of today, we can enumerate many quantization methods developped by the community for various usecases. We offer to the community the possibility to easily run these methods and it is up to the users to decide on which quantization method to go with for their usecases, as all of them have their own pros and cons. See the table below for more details.

<Tip>

Interested in adding a new quantization method to Transformers? Read the [HfQuantizer](./contribute) guide to learn how!

</Tip>

## Overview

Below is a brief overview of the supported quantization method with their characteristics:

| Quantization method                 | CPU | GPU | Metal (Apple Silicon) | torch.compile() support | Number of bits | Supports fine-tuning (through PEFT) | Support Quantization-Aware Training | Serializable | 🤗 transformers support | Link to library                             |
|-------------------------------------|-----|-----|-----------------------|-------------------------|----------------|-------------------------------------|-------------------------------------|--------------|------------------------|---------------------------------------------|
| AQLM                                | ✅   | ✅   | ❌                     | ?                       | 1 / 2          | ✅                                   | ❌                                   | ✅            | ✅                      | https://github.com/Vahe1994/AQLM            |
| AWQ (Activation Aware Quantization) | ❌   | ✅   | ❌                     | ?                       | 4              | ✅                                   | ❌                                   | ✅            | ✅                      | https://github.com/casper-hansen/AutoAWQ    |
| bitsandbytes                        | ❌   | ✅   | ❌                     | ❌                       | 4 / 8          | ✅                                   | ❌                                   | ✅            | ✅                      | https://github.com/TimDettmers/bitsandbytes |
| EETQ                                | ❌   | ✅   | ❌                     | ?                       | 8              | ✅                                   | ❌                                   | ✅            | ✅                      | https://github.com/NetEase-FuXi/EETQ        |
| GGUF / GGML (llama.cpp)             | ✅   | ✅   | ✅                     | ❌                       | 1 - 8          | ❌                                   | ❌                                   | ✅            | ❌                      | https://github.com/ggerganov/llama.cpp      |
| HQQ                                 | ❌   | ✅   | ❌                     | ✅                       | 1 - 8          | ✅                                   | ❌                                   | ❌            | ✅                      | https://github.com/mobiusml/hqq/            |
| Quanto                              | ✅   | ✅   | ✅                     | ✅                       | 2 / 4 / 8      | ❌                                   | ✅                                   | ❌            | ✅                      | https://github.com/huggingface/quanto       |