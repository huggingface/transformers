# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Transformations Utilities for Time Series Transformers. """

from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, IterableSlice, PseudoShuffled
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.util import IterableDataset
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    RenameFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler


PREDICTION_INPUT_NAMES = [
    "static_categorical_features",
    "static_real_features",
    "past_time_features",
    "past_values",
    "past_observed_mask",
    "future_time_features",
]

TRAIN_VAL_INPUT_NAMES = PREDICTION_INPUT_NAMES + ["future_values", "future_observed_mask"]


@lru_cache(10_000)
def _as_period(val, freq):
    return pd.Period(val, freq)


def transform_data(batch, freq, log1p_transform=False):
    batch[FieldName.START] = [_as_period(entry, freq) for entry in batch[FieldName.START]]
    if log1p_transform:
        batch[FieldName.TARGET] = np.log1p(batch[FieldName.TARGET])
    return batch


def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: use static features if available, if not add dummy values
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
            if not config.num_static_categorical_features > 0
            else []
        )
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
            if not config.num_static_real_features > 0
            else []
        )
        # step 3: convert the data to NumPy (potentially not needed)
        + [
            AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1, dtype=int),
            AsNumpyArray(field=FieldName.FEAT_STATIC_REAL, expected_ndim=1),
            AsNumpyArray(
                field=FieldName.TARGET,
                # in the following line, we add 1 for the time dimension
                expected_ndim=config.input_size,
            ),
            # step 4: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(target_field=FieldName.TARGET, output_field=FieldName.OBSERVED_VALUES),
            # step 5: add temporal features based on freq of the dataset
            # month of year in this case
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 6: add another temporal feature (just a single number per time step)
            # tells the model where in the life the value of the time series is
            # kind of running counter which is log transformed
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 7: vertically stack all the temporal features
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + ([FieldName.FEAT_DYNAMIC_REAL] if config.num_dynamic_real_features > 0 else []),
            ),
            # step 8: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )


def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler or ExpectedNumInstanceSampler(num_instances=1.0, min_future=config.prediction_length),
        "validation": validation_sampler or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )


def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    **kwargs,
) -> Iterable:
    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)

    # we initialize a Training instance splitter
    instance_splitter = create_instance_splitter(config, "train") + SelectFields(TRAIN_VAL_INPUT_NAMES)

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the transformed time series of a dataset)
    # randomly from within the values of the time series and return another iterator.
    training_instances = instance_splitter.apply(
        Cyclic(transformed_data)
        if shuffle_buffer_length is None
        else PseudoShuffled(Cyclic(transformed_data), shuffle_buffer_length=shuffle_buffer_length)
    )

    # from the training instances iterator we now return a Dataloader which will
    # continue to sample random windows for as long as it is called
    # to return batch_size of the appropriate tensors ready for training!
    return IterableSlice(
        iter(DataLoader(IterableDataset(training_instances), batch_size=batch_size, **kwargs)),
        length=num_batches_per_epoch,
    )


def create_validation_dataloader(freq, config, data, batch_size, **kwargs):
    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)

    # we initialize a Validation instance splitter
    instance_splitter = create_instance_splitter(config, "validation") + SelectFields(TRAIN_VAL_INPUT_NAMES)
    validation_instances = instance_splitter.apply(transformed_data, is_train=True)

    return DataLoader(IterableDataset(validation_instances), batch_size=batch_size, **kwargs)


def create_test_dataloader(config: PretrainedConfig, freq, data, batch_size: int, **kwargs):
    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_splitter = create_instance_splitter(config, "test") + SelectFields(PREDICTION_INPUT_NAMES)

    # we apply the transformations in test mode
    testing_instances = instance_splitter.apply(transformed_data, is_train=False)

    # This returns a Dataloader which will go over the dataset once.
    return DataLoader(IterableDataset(testing_instances), batch_size=batch_size, **kwargs)
