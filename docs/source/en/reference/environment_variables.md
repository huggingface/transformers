<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Environment Variables

## HF_ENABLE_PARALLEL_DOWNLOADING

By default this is disabled. Enables the parallel downloading of models with sharded weight files. Can decrease the time to load large models significantly, often times producing _speed ups of greater than 50%_.

Can be set to a string equal to `"false"` or `"true"`. e.g. `os.environ["HF_ENABLE_PARALLEL_DOWNLOADING"] = "true"`

While downloading is already parallelized at the file level when `HF_HUB_ENABLE_HF_TRANSFER` is enabled, `HF_ENABLE_PARALLEL_DOWNLOADING` parallelizes the number of files that can be concurrently downloaded. Which can greatly speed up downloads if the machine you're using can handle it in terms of network and IO bandwidth.

e.g. here's a comparison for `facebook/opt-30b` on an AWS EC2 `g4dn.metal`:

- `HF_HUB_ENABLE_HF_TRANSFER` enabled, `HF_ENABLE_PARALLEL_DOWNLOADING` disabled

  - ~45s download

- `HF_HUB_ENABLE_HF_TRANSFER` enabled, `HF_ENABLE_PARALLEL_DOWNLOADING` enabled
  - ~12s download

To fully saturate a machine capable of massive network bandwidth, set `HF_ENABLE_PARALLEL_DOWNLOADING="true"` and `HF_HUB_ENABLE_HF_TRANSFER="1"`

_Note, you will want to profile your code before committing to using this environment variable, this will not produce speed ups for smaller models._

```py
import os
from transformers import pipeline

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true" # enable parallized pool of downloader threads

model = pipeline(task="text-generation", model="facebook/opt-30b", device_map="auto")
```

## HF_PARALLEL_DOWNLOADING_WORKERS

Determines how many threads should be used when the parallel downloading of model weight shards is enabled.

Default is `8` although less may run if the number of shard files is less than the number of workers. i.e. it takes the min() of HF_PARALLEL_DOWNLOADING_WORKERS and the number of shard files to download.

e.g. if there are 2 shard files, but 8 workers are specified, only two workers will spawn.

```py
import os
from transformers import pipeline

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true" # enable parallized pool of downloader threads
os.environ["HF_PARALLEL_DOWNLOADING_WORKERS"] = "12" # Specify a non default number of workers

model = pipeline(task="text-generation", model="facebook/opt-30b", device_map="auto")
```
