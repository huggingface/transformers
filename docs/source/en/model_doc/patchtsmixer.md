<!--Copyright 2023 IBM and HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-06-14 and added to Hugging Face Transformers on 2023-12-05 and contributed by [ajati](https://huggingface.co/ajati) and [vijaye12](https://huggingface.co/vijaye12).*

# PatchTSMixer

[PatchTSMixer](https://huggingface.co/papers/2306.09364) is a lightweight neural architecture for multivariate time series forecasting that replaces Transformers with multi-layer perceptrons (MLPs) applied to patched time series. It introduces online reconciliation heads to explicitly model hierarchical and channel correlations, along with a hybrid channel modeling approach and a simple gating mechanism to manage noisy interactions and improve generalization. The model achieves strong performance with minimal computational cost, outperforming both state-of-the-art MLP and Transformer models by 8–60%, and Patch-Transformer benchmarks by 1–2%, while reducing memory and runtime by 2–3×. Its modular design supports both supervised and masked self-supervised learning, positioning it as a flexible building block for time-series foundation models.

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

model = PatchTSTForPrediction.from_pretrained("ibm-granite/granite-timeseries-patchtsmixer", dtype="auto")

outputs = model(past_values=batch["past_values"])
prediction_outputs = outputs.prediction_outputs
```

</hfoption>
</hfoptions>

## Usage tips

- Use the model for time series classification and regression. See [`PatchTSMixerForTimeSeriesClassification`] and [`PatchTSMixerForRegression`] classes.

## PatchTSMixerConfig

[[autodoc]] PatchTSMixerConfig

## PatchTSMixerModel

[[autodoc]] PatchTSMixerModel
    - forward

## PatchTSMixerForPrediction

[[autodoc]] PatchTSMixerForPrediction
    - forward

## PatchTSMixerForTimeSeriesClassification

[[autodoc]] PatchTSMixerForTimeSeriesClassification
    - forward

## PatchTSMixerForPretraining

[[autodoc]] PatchTSMixerForPretraining
    - forward

## PatchTSMixerForRegression

[[autodoc]] PatchTSMixerForRegression
    - forward

