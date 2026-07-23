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

# Candle

[Candle](https://github.com/huggingface/candle) is a machine learning framework providing native Rust implementations of Transformers models. It natively supports [safetensors](https://huggingface.co/docs/safetensors/en/index) to load Transformers models directly.

```rust
/// load model config
let config: Config = 
    serde_json::from_reader(std::fs::File::open(config_filename)?)?;

/// load safetensors and memory-maps them
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)?
};

/// materialize tensors from VarBuilder into model class
let model = Model::new(args.use_flash_attn, &config, vb)?;
```

## Transformers integration

1. The [hf-hub](https://github.com/huggingface/hf-hub) crate checks your local [Hugging Face cache](../installation#cache-directory) for a model. If it isn't there, it downloads model weights and configs from the Hub.
2. [VarBuilder](https://github.com/huggingface/candle/blob/f526033db7ea880c7189628a2dc00e3e2008a9e7/candle-nn/src/var_builder.rs#L38) lazily loads the safetensor files. It maps state-dict key names to Rust structs representing model layers. This mirrors how Transformers organizes its weights.
3. Candle parses `config.json` to extract model metadata and instantiates the matching Rust model class with weights from `VarBuilder`.

## Resources

- [Candle](https://github.com/huggingface/candle) documentation