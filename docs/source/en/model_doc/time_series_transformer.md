<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Time Series Transformer

<Tip>

This is a recently introduced model so the API hasn't been tested extensively. There may be some bugs or slight
breaking changes to fix it in the future. If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title).

</Tip>

## Overview

The Time Series Transformer model is a vanilla encoder-decoder Transformer for time series forecasting.

Tips:

- Similar to other models in the library, [`TimeSeriesTransformerModel`] is the raw Transformer without any head on top, and [`TimeSeriesTransformerForPrediction`]
adds a distribution head on top of the former, which can be used for time-series forecasting. Note that this is a so-called probabilistic forecasting model, not a
point forecasting model. This means that the model learns a distribution, from which one can sample. The model doesn't directly output values.
- [`TimeSeriesTransformerForPrediction`] consists of 2 blocks: an encoder, which takes a `context_length` of time series values as input (called `past_values`),
and a decoder, which predicts a `prediction_length` of time series values into the future (called `future_values`). During training, one needs to provide
pairs of (`past_values` and `future_values`) to the model.
- In addition to the raw (`past_values` and `future_values`), one typically provides additional features to the model. These can be the following:
    - `past_time_features`: temporal features which the model will add to `past_values`. These serve as "positional encodings" for the Transformer encoder.
    Examples are "day of the month", "month of the year", etc. as scalar values (and then stacked together as a vector).
    e.g. if a given time-series value was obtained on the 11th of August, then one could have [11, 8] as time feature vector (11 being "day of the month", 8 being "month of the year").
    - `future_time_features`: temporal features which the model will add to `future_values`. These serve as "positional encodings" for the Transformer decoder.
    Examples are "day of the month", "month of the year", etc. as scalar values (and then stacked together as a vector).
    e.g. if a given time-series value was obtained on the 11th of August, then one could have [11, 8] as time feature vector (11 being "day of the month", 8 being "month of the year").
    - `static_categorical_features`: categorical features which are static over time (i.e., have the same value for all `past_values` and `future_values`).
    An example here is the store ID or region ID that identifies a given time-series.
    Note that these features need to be known for ALL data points (also those in the future).
    - `static_real_features`: real-valued features which are static over time (i.e., have the same value for all `past_values` and `future_values`).
    An example here is the image representation of the product for which you have the time-series values (like the [ResNet](resnet) embedding of a "shoe" picture,
    if your time-series is about the sales of shoes).
    Note that these features need to be known for ALL data points (also those in the future).
- The model is trained using "teacher-forcing", similar to how a Transformer is trained for machine translation. This means that, during training, one shifts the
`future_values` one position to the right as input to the decoder, prepended by the last value of `past_values`. At each time step, the model needs to predict the
next target. So the set-up of training is similar to a GPT model for language, except that there's no notion of `decoder_start_token_id` (we just use the last value
of the context as initial input for the decoder).
- At inference time, we give the final value of the `past_values` as input to the decoder. Next, we can sample from the model to make a prediction at the next time step,
which is then fed to the decoder in order to make the next prediction (also called autoregressive generation).


This model was contributed by [kashif](https://huggingface.co/kashif).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- Check out the Time Series Transformer blog-post in HuggingFace blog: [Probabilistic Time Series Forecasting with 🤗 Transformers](https://huggingface.co/blog/time-series-transformers)


## TimeSeriesTransformerConfig

[[autodoc]] TimeSeriesTransformerConfig


## TimeSeriesTransformerModel

[[autodoc]] TimeSeriesTransformerModel
    - forward


## TimeSeriesTransformerForPrediction

[[autodoc]] TimeSeriesTransformerForPrediction
    - forward
