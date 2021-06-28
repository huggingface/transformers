<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Troubleshooting

This document is to help find solutions for common problems.

## Firewalled environments

Some cloud and intranet setups have their GPU instances firewalled to the outside world, so if your script is trying to download model weights or datasets it will first hang and then timeout with an error message like:

```
ValueError: Connection error, and we cannot find the requested files in the cached path.
Please try again or make sure your Internet connection is on.
```

One possible solution in this situation is to use the ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode).
