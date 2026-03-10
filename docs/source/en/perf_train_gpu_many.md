<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# N-D parallelism

N-dimensional parallelism uses multiple parallelism strategies at the same time. Each dimension is a different way to split work across GPUs, such as splitting on the data, individual model tensors, or model layers.

| parallelism | description |
|---|---|
| DP + TP | data + tensor parallelism |
| DP + TP + PP | data + tensor + pipeline parallelism |
| DP + TP + PP + SP | data + tensor + pipeline + sequence parallelism |
| DP + TP + PP + SP + EP | data + tensor + pipeline + sequence parallelism + expert parallelism |

