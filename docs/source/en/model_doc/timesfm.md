<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-10-14 and added to Hugging Face Transformers on 2025-04-16.*

# TimesFM

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model proposed in [A decoder-only foundation model for time-series forecasting](https://huggingface.co/papers/2310.10688) by Abhimanyu Das, Weihao Kong, Rajat Sen, and  Yichen Zhou. It is a decoder only model that uses non-overlapping patches of time-series data as input and outputs some output patch length prediction in an autoregressive fashion.

The abstract from the paper is the following:

*Motivated by recent advances in large language models for Natural Language Processing (NLP), we design a time-series foundation model for forecasting whose out-of-the-box zero-shot performance on a variety of public datasets comes close to the accuracy of state-of-the-art supervised forecasting models for each individual dataset. Our model is based on pretraining a patched-decoder style attention model on a large time-series corpus, and can work well across different forecasting history lengths, prediction lengths and temporal granularities.*

This model was contributed by [kashif](https://huggingface.co/kashif).
The original code can be found [here](https://github.com/google-research/timesfm).

To use the model:

```python
import numpy as np
import torch
from transformers import TimesFmModelForPrediction


model = TimesFmModelForPrediction.from_pretrained(
    "google/timesfm-2.0-500m-pytorch",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto"
)


 # Create dummy inputs
forecast_input = [
    np.sin(np.linspace(0, 20, 100)),
    np.sin(np.linspace(0, 20, 200)),
    np.sin(np.linspace(0, 20, 400)),
]
frequency_input = [0, 1, 2]

# Convert inputs to sequence of tensors
forecast_input_tensor = [
    torch.tensor(ts, dtype=torch.bfloat16).to(model.device)
    for ts in forecast_input
]
frequency_input_tensor = torch.tensor(frequency_input, dtype=torch.long).to(model.device)

# Get predictions from the pre-trained model
with torch.no_grad():
    outputs = model(past_values=forecast_input_tensor, freq=frequency_input_tensor, return_dict=True)
    point_forecast_conv = outputs.mean_predictions.float().cpu().numpy()
    quantile_forecast_conv = outputs.full_predictions.float().cpu().numpy()
```

## Forecasting with Covariates

TimesFM supports forecasting with external covariates using batched in-context regression. This allows you to incorporate additional information such as weather data, economic indicators, or business metrics to improve forecast accuracy.

The model supports four types of covariates:

- **Dynamic Numerical**: Time-varying numerical features (e.g., temperature, price)
- **Dynamic Categorical**: Time-varying categorical features (e.g., day of week, season)
- **Static Numerical**: Time-invariant numerical features (e.g., store size, population)
- **Static Categorical**: Time-invariant categorical features (e.g., region, store type)

### Basic Example

```python
import numpy as np
import torch
from transformers import TimesFmModelForPrediction

# Load the model
model = TimesFmModelForPrediction.from_pretrained(
    "google/timesfm-2.0-500m-pytorch",
    dtype=torch.bfloat16,
    device_map="auto"
)

# Prepare historical time series data (ice cream sales example)
# Match the model's dtype and device for proper compatibility
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
past_sales = [
    torch.tensor([45, 52, 48, 55, 61, 58, 62, 59, 56, 53], dtype=dtype, device=device),  # Store 1
    torch.tensor([38, 42, 39, 46, 48, 45, 49, 47, 44, 41], dtype=dtype, device=device),  # Store 2
]

# Prepare covariates (context + future)
context_len = 10
horizon_len = 5
total_len = context_len + horizon_len

# Dynamic numerical covariates (temperature affects ice cream sales)
temperature_store1 = [22, 25, 23, 28, 31, 29, 32, 30, 27, 24,  # context
                      26, 29, 31, 33, 30]                      # future (horizon)
temperature_store2 = [20, 23, 21, 26, 29, 27, 30, 28, 25, 22,  # context
                      24, 27, 29, 31, 28]                      # future (horizon)

dynamic_numerical = {
    "temperature": [temperature_store1, temperature_store2]
}

# Dynamic categorical covariates (day of week effect)
dynamic_categorical = {
    "weekday": [
        [1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1],  # Store 1: Mon=1, Sun=0
        [1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1],  # Store 2
    ]
}

# Static covariates (store characteristics)
static_numerical = {
    "store_size": [150.0, 120.0],  # sq ft (hundreds)
}

static_categorical = {
    "store_type": ["mall", "street"],
    "region": ["north", "south"],
}

# Generate forecasts with covariates
with torch.no_grad():
    outputs = model.forecast_with_covariates(
        past_values=past_sales,
        dynamic_numerical_covariates=dynamic_numerical,
        dynamic_categorical_covariates=dynamic_categorical,
        static_numerical_covariates=static_numerical,
        static_categorical_covariates=static_categorical,
        ridge=0.1,  # Ridge regularization for stability
    )

# Extract results
combined_forecast = outputs.combined_predictions  # TimesFM + XReg predictions
xreg_forecast = outputs.xreg_predictions          # XReg-only predictions
timesfm_forecast = outputs.mean_predictions       # TimesFM-only predictions

print(f"Combined forecast shape: {combined_forecast.shape}")  # [2, 5]
print(f"Store 1 combined forecast: {combined_forecast[0].cpu().numpy()}")
print(f"Store 2 combined forecast: {combined_forecast[1].cpu().numpy()}")
```

### Advanced Example: Electricity Price Forecasting

This example demonstrates forecasting electricity prices with multiple covariates, inspired by electricity price forecasting (EPF) scenarios:

```python
import numpy as np
import torch
from transformers import TimesFmModelForPrediction

# Load model
model = TimesFmModelForPrediction.from_pretrained(
    "google/timesfm-2.0-500m-pytorch", 
    dtype=torch.float32
)

# Historical electricity prices (48 hours of context)
np.random.seed(42)
context_hours = 48
horizon_hours = 24
total_hours = context_hours + horizon_hours

# Create realistic price patterns for 3 regions
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
past_prices = []
for region in range(3):
    # Daily pattern: higher during day, lower at night
    daily_pattern = 50 + 20 * np.sin(2 * np.pi * np.arange(context_hours) / 24)
    # Add regional base price and noise
    regional_base = 40 + region * 10
    noise = np.random.randn(context_hours) * 5
    prices = regional_base + daily_pattern + noise
    past_prices.append(torch.tensor(prices, dtype=dtype, device=device))

# Dynamic numerical covariates
load_demand = []
temperature = []
renewable_share = []

for region in range(3):
    # Electricity load (MW) - main price driver
    base_load = 1000 + 300 * np.sin(2 * np.pi * np.arange(total_hours) / 24)
    regional_load = base_load + region * 100 + np.random.randn(total_hours) * 50
    load_demand.append(regional_load.tolist())

    # Temperature (affects demand)
    temp_pattern = 20 + 10 * np.sin(2 * np.pi * np.arange(total_hours) / (24 * 30))
    temp_noise = np.random.randn(total_hours) * 3
    temperature.append((temp_pattern + temp_noise).tolist())

    # Renewable energy share (affects pricing)
    renewable = np.clip(0.3 + 0.2 * np.random.randn(total_hours), 0.1, 0.8)
    renewable_share.append(renewable.tolist())

dynamic_numerical = {
    "load_mw": load_demand,
    "temperature": temperature, 
    "renewable_share": renewable_share,
}

# Dynamic categorical covariates 
dynamic_categorical = {
    "hour": [
        [i % 24 for i in range(total_hours)]  # Hour of day: 0-23
        for _ in range(3)
    ],
    "day_type": [
        ["weekday" if (i // 24) % 7 < 5 else "weekend" for i in range(total_hours)]
        for _ in range(3)
    ],
}

# Static covariates (market characteristics)
static_numerical = {
    "market_capacity_mw": [5000.0, 4500.0, 6000.0],
    "transmission_capacity": [800.0, 700.0, 900.0],
}

static_categorical = {
    "market_type": ["competitive", "regulated", "competitive"],
    "primary_fuel": ["gas", "coal", "nuclear"],
}

# Forecast with covariates
with torch.no_grad():
    outputs = model.forecast_with_covariates(
        past_values=past_prices,
        dynamic_numerical_covariates=dynamic_numerical,
        dynamic_categorical_covariates=dynamic_categorical,
        static_numerical_covariates=static_numerical,
        static_categorical_covariates=static_categorical,
        xreg_mode="xreg + timesfm",  # Fit XReg first, then TimesFM on residuals
        ridge=0.5,  # Higher ridge for stability with many covariates
    )

price_forecasts = outputs.combined_predictions
print(f"24-hour price forecasts for {len(price_forecasts)} regions:")
for i, forecast in enumerate(price_forecasts):
    print(f"Region {i+1}: ${forecast.mean():.2f}/MWh (avg)")
```

### XReg Modes

TimesFM supports two modes for combining TimesFM and external regression (XReg) predictions:

1. **"xreg + timesfm"** (default): Fit linear model on targets first, then forecast residuals with TimesFM
2. **"timesfm + xreg"**: Forecast with TimesFM first, then fit a linear model on residuals

```python
# Compare different modes
modes = ["xreg + timesfm", "timesfm + xreg"]

for mode in modes:
    with torch.no_grad():
        outputs = model.forecast_with_covariates(
            past_values=past_sales,
            dynamic_numerical_covariates={"temperature": temperature_data},
            xreg_mode=mode,
            ridge=0.1,
        )
    print(f"{mode}: {outputs.combined_predictions[0][:3].cpu().numpy()}")
```

### Key Parameters

- **`ridge`**: Ridge regularization parameter (0.0-1.0). Higher values provide more stability with many covariates
- **`normalize_xreg_target_per_input`**: Whether to normalize targets per input series (default: True)
- **`xreg_mode`**: How to combine TimesFM and XReg predictions
- **`truncate_negative`**: Whether to truncate negative predictions for non-negative data

The covariate forecasting leverages batched in-context regression to efficiently process multiple time series with external information, enabling more accurate forecasts for complex real-world scenarios.

## TimesFmConfig

[[autodoc]] TimesFmConfig

## TimesFmModel

[[autodoc]] TimesFmModel
    - forward

## TimesFmModelForPrediction

[[autodoc]] TimesFmModelForPrediction
    - forward
    - forecast_with_covariates
