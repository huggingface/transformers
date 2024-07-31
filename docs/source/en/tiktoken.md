<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
``
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Tiktoken and interaction with Transformers
[Tiktoken](https://github.com/openai/tiktoken) is an efficient tokenizer developed by OpenAI. It is optimized for speed 
and minimal memory usage.

Support for tiktoken model files is seamlessly integrated in 🤗 transformers when loading models 
`from_pretrained` with a `tokenizer.model` tiktoken file on the Hub, which automatically converted into our 
[fast tokenizer](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast). 
All necessary components, including model weights, tokenizer settings, and configuration metadata, are 
encapsulated within a single file, simplifying model distribution and deployment. This means there is no need for 
separate files for the model, tokenizer, and configuration settings.

### Supported model architectures

- #TODO: add model architectures which has tiktoken file on hub!

## Example usage
 
In order to load `tiktoken` files in `transformers`, ensure that the `tokenizer.model` file is a tiktoken file and it 
will automatically be loaded when loading `from_pretrained`. Here is how one would load a tokenizer and a model, which 
 can be loaded from the exact same file:

```py
from transformers import AutoTokenizer

model_id = #TODO: Add model id

tokenizer = AutoTokenizer.from_pretrained(model_id)
```
