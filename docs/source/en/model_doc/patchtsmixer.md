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

# PatchTSMixer

## Overview

The PatchTSMixer model was proposed in [TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting](https://arxiv.org/pdf/2306.09364.pdf) by Vijay Ekambaram, Arindam Jati, Nam Nguyen, Phanwadee Sinthong and Jayant Kalagnanam.


PatchTSMixer is a lightweight time-series modeling approach based on the MLP-Mixer architecture. In this HuggingFace implementation, we provide PatchTSMixer's capabilities to effortlessly facilitate lightweight mixing across patches, channels, and hidden features for effective multivariate time-series modeling. It also supports various attention mechanisms starting from simple gated attention to more complex self-attention blocks that can be customized accordingly. The model can be pretrained and subsequently used for various downstream tasks such as forecasting, classification and regression.


The abstract from the paper is the following:

*TSMixer is a lightweight neural architecture exclusively composed of multi-layer perceptron (MLP) modules designed for multivariate forecasting and representation learning on patched time series. Our model draws inspiration from the success of MLP-Mixer models in computer vision. We demonstrate the challenges involved in adapting Vision MLP-Mixer for time series and introduce empirically validated components to enhance accuracy. This includes a novel design paradigm of attaching online reconciliation heads to the MLP-Mixer backbone, for explicitly modeling the time-series properties such as hierarchy and channel-correlations. We also propose a Hybrid channel modeling approach to effectively handle noisy channel interactions and generalization across diverse datasets, a common challenge in existing patch channel-mixing methods. Additionally, a simple gated attention mechanism is introduced in the backbone to prioritize important features. By incorporating these lightweight components, we significantly enhance the learning capability of simple MLP structures, outperforming complex Transformer models with minimal computing usage. Moreover, TSMixer's modular design enables compatibility with both supervised and masked self-supervised learning methods, making it a promising building block for time-series Foundation Models. TSMixer outperforms state-of-the-art MLP and Transformer models in forecasting by a considerable margin of 8-60%. It also outperforms the latest strong benchmarks of Patch-Transformer models (by 1-2%) with a significant reduction in memory and runtime (2-3X).*

This model was contributed by [ajati](https://huggingface.co/ajati), [vijaye12](https://huggingface.co/vijaye12), 
[gsinthong](https://huggingface.co/gsinthong), [namctin](https://huggingface.co/namctin),
[wmgifford](https://huggingface.co/wmgifford), [kashif](https://huggingface.co/kashif).

## Usage example

The code snippet below shows how to randomly initialize a PatchTSMixer model. The model is compatible with the [Trainer API](../trainer.md).

```python

from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction
from transformers import Trainer, TrainingArguments,


config = PatchTSMixerConfig(context_length = 512, prediction_length = 96)
model = PatchTSMixerForPrediction(config)
trainer = Trainer(model=model, args=training_args, 
            train_dataset=train_dataset,
            eval_dataset=valid_dataset)
trainer.train()
results = trainer.evaluate(test_dataset)
```

## Usage tips

The model can also be used for time series classification and time series regression. See the respective [`PatchTSMixerForTimeSeriesClassification`] and [`PatchTSMixerForRegression`] classes.

## Resources

- A blog post explaining PatchTSMixer in depth can be found [here](https://huggingface.co/blog/patchtsmixer). The blog can also be opened in Google Colab.

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