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
*This model was released on 2022-11-27 and added to Hugging Face Transformers on 2023-11-13 and contributed by [namctin](https://huggingface.co/namctin), [gsinthong](https://huggingface.co/gsinthong), [diepi](https://huggingface.co/diepi), [vijaye12](https://huggingface.co/vijaye12), [wmgifford](https://huggingface.co/wmgifford), and [kashif](https://huggingface.co/kashif).*

# PatchTST

[https://huggingface.co/papers/2211.14730](PatchTST) proposes an efficient Transformer-based model for multivariate time series forecasting and self-supervised representation learning. The model segments time series into subseries-level patches, which are used as input tokens for the Transformer. Each channel, representing a single univariate time series, shares the same embedding and Transformer weights. This design retains local semantic information, reduces computation and memory usage, and allows the model to consider longer historical data. PatchTST significantly improves long-term forecasting accuracy compared to state-of-the-art Transformer-based models. Additionally, the model achieves excellent fine-tuning performance in self-supervised pre-training tasks and demonstrates superior forecasting accuracy when transferring pre-trained representations across datasets.

<hfoptions id="usage">
<hfoption id="PatchTSTForPrediction">

```py
import torch
from huggingface_hub import hf_hub_download
from transformers import PatchTSTConfig, PatchTSTForPrediction

file = hf_hub_download(
    repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = PatchTSTForPrediction.from_pretrained("namctin/patchtst_etth1_forecast", dtype="auto")

outputs = model(past_values=batch["past_values"])
prediction_outputs = outputs.prediction_outputs
```

</hfoption>
</hfoptions>

## PatchTSTConfig

[[autodoc]] PatchTSTConfig

## PatchTSTModel

[[autodoc]] PatchTSTModel
    - forward

## PatchTSTForPrediction

[[autodoc]] PatchTSTForPrediction
    - forward

## PatchTSTForClassification

[[autodoc]] PatchTSTForClassification
    - forward

## PatchTSTForPretraining

[[autodoc]] PatchTSTForPretraining
    - forward

## PatchTSTForRegression

[[autodoc]] PatchTSTForRegression
    - forward

