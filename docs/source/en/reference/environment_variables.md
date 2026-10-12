<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Environment variables

Boolean variables accept `1`, `on`, `yes`, or `true` in any capitalization. Anything else, including unset, is false.

Most of these variables are read once, when Transformers is imported. Set them in your shell before launching Python, or with [os.environ](https://docs.python.org/3/library/os.html#os.environ) before the import. Changing them afterwards has no effect.

```bash
TRANSFORMERS_VERBOSITY=info python run.py
```

Caching and Hub access are configured by the `HF_*` variables [huggingface_hub](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) owns, such as `HF_HOME` and `HF_HUB_OFFLINE`. See [Installation](../installation#offline-mode) for the offline workflow.

## Logging

Transformers logs at `warning`. [Logging](../main_classes/logging) documents the equivalent Python API, including [`logging.set_verbosity`].

| Variable | Values | Description |
|---|---|---|
| `TRANSFORMERS_VERBOSITY` | `detail`, `debug`, `info`, `warning`, `error`, `critical` | Library log level. `detail` matches `debug` and adds the filename and line number to each message. An unrecognized value warns and falls back to `warning`. |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | boolean | Silences advisory warnings, the best-practice hints that don't indicate a problem. |

## Model loading

Checkpoint shards load in parallel by default.

| Variable | Values | Description |
|---|---|---|
| `HF_DEACTIVATE_ASYNC_LOAD` | boolean | Loads shards on the main thread instead of a thread pool. Slower, but peak memory is easier to reason about. Loading is already sequential when a `device_map` offloads to disk or when quantizing on the fly, so this only affects otherwise parallel loads. |
| `DISABLE_SAFETENSORS_CONVERSION` | boolean | Stops Transformers from asking the Hub conversion bot for a safetensors version of a repository that ships only PyTorch `.bin` weights. Worth setting in CI, where the request adds latency and network flakiness. |

## Kernels and backends

Optimized kernels are downloaded from the Hub when [kernels](https://github.com/huggingface/kernels) is installed and the hardware supports them. Disable them to isolate a numerical difference or a crash.

| Variable | Values | Description |
|---|---|---|
| `USE_HUB_KERNELS` | boolean, enabled by default | Set to a false value to fall back to the reference implementations. |
| `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR` | `1` | Forces the Triton fallback instead of DeepGEMM for FP8 linear layers in finegrained FP8 quantization. |
| `TRANSFORMERS_DISABLE_TORCH_CHECK` | `1` | Skips the minimum PyTorch version check at import. |

> [!WARNING]
> `TRANSFORMERS_DISABLE_TORCH_CHECK` removes a guard rather than changing behavior. On an unsupported PyTorch version you get import errors or silently wrong results instead of a clear message.
