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
*This model was released on 2020-12-14 and added to Hugging Face Transformers on 2023-03-08 and contributed by [elisim](https://huggingface.co/elisim) and [kashif](https://huggingface.co/kashif).*

# Informer

[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://huggingface.co/papers/2012.07436) addresses the challenges of long sequence time-series forecasting (LSTF) by introducing a ProbSparse self-attention mechanism that reduces time complexity and memory usage to O(L logL). It also employs self-attention distillation to focus on significant attention points and uses a generative style decoder for efficient long-sequence predictions. These features enable Informer to outperform existing methods on large-scale datasets.

<hfoptions id="usage">
<hfoption id="InformerForPrediction">

```py
import torch
from huggingface_hub import hf_hub_download
from transformers import InformerForPrediction

file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly", dtype="auto")
outputs = model.generate(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"],
    static_real_features=batch["static_real_features"],
    future_time_features=batch["future_time_features"],
)

mean_prediction = outputs.sequences.mean(dim=1)
```

</hfoption>
</hfoptions>

## InformerConfig

[[autodoc]] InformerConfig

## InformerModel

[[autodoc]] InformerModel
    - forward

## InformerForPrediction

[[autodoc]] InformerForPrediction
    - forward

