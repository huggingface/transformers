<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a C/C++ inference engine for deploying large language models locally. It's lightweight and doesn't require Python, CUDA, or other heavy server infrastructure. llama.cpp uses the [GGUF](https://huggingface.co/blog/ngxson/common-ai-model-formats#gguf) file format. GGUF supports quantized model weights and memory-mapping to reduce memory bandwidth on your device.

> [!TIP]
> Browse the [Hub](https://huggingface.co/models?apps=llama.cpp&sort=trending) for models already available in GGUF format.

llama.cpp can convert and run Transformers models as standalone C++ executables with the [convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) script.

```bash
python3 convert_hf_to_gguf.py ./models/openai/gpt-oss-20b
```

The conversion process works as follows.

1. The script loads the model configuration with [`AutoConfig.from_pretrained`] and extracts the 
vocabulary with [`AutoTokenizer.from_pretrained`].
2. Based on the config's architecture field, the script selects a converter class from its internal registry. The registry maps Transformers architecture names (like [`LlamaForCausalLM`]) to corresponding converter classes.
3. The converter class extracts config parameters, maps Transformers tensor names to GGUF tensor names, transforms tensors, and packages the vocabulary.
4. The output is a single GGUF file containing the model weights, tokenizer, and metadata.

Deploy the model locally from the command line with [llama-cli](https://github.com/ggml-org/llama.cpp/tree/master#llama-cli) or start a web UI with [llama-server](https://github.com/ggml-org/llama.cpp/tree/master#llama-server). Add the `-hf` flag to indicate the model is from the Hub.

<hfoptions id="deploy">
<hfoption id="llama-cli">

```bash
llama-cli -hf openai/gpt-oss-20b
```

</hfoption>
<hfoption id="llama-server">

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF
```

</hfoption>
</hfoptions>

## Resources

- [llama.cpp](https://github.com/ggml-org/llama.cpp) documentation
- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml) blog post