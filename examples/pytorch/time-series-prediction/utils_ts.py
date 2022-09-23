# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler


@lru_cache(10_000)
def _as_period(val, freq):
    return pd.Period(val, freq)


def transform_start_field(batch, freq):
    batch[FieldName.START] = [_as_period(entry, freq) for entry in batch[FieldName.START]]
    return batch


def create_transformation(freq, config) -> Transformation:
    remove_field_names = []
    if config.num_feat_static_real == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_feat_dynamic_real == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

    return Chain(
        [RemoveFields(field_names=remove_field_names)]
        + ([SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])] if not config.num_feat_static_cat > 0 else [])
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
            if not config.num_feat_static_real > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=1,
                dtype=int,
            ),
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_REAL,
                expected_ndim=1,
            ),
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=config.input_size,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + ([FieldName.FEAT_DYNAMIC_REAL] if config.num_feat_dynamic_real > 0 else []),
            ),
        ]
    )


def create_instance_splitter(
    config,
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
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_seq),
        future_length=config.prediction_length,
        time_series_fields=[
            FieldName.FEAT_TIME,
            FieldName.OBSERVED_VALUES,
        ],
    )


def create_training_data_loader(
    freq,
    config,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        FieldName.FEAT_STATIC_CAT,
        FieldName.FEAT_STATIC_REAL,
        "past_" + FieldName.FEAT_TIME,
        "past_" + FieldName.TARGET,
        "past_" + FieldName.OBSERVED_VALUES,
        "future_" + FieldName.FEAT_TIME,
    ]

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_" + FieldName.TARGET,
        "future_" + FieldName.OBSERVED_VALUES,
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)

    instance_splitter = create_instance_splitter(config, "train") + SelectFields(TRAINING_INPUT_NAMES)

    training_instances = instance_splitter.apply(
        Cyclic(transformed_data)
        if shuffle_buffer_length is None
        else PseudoShuffled(
            Cyclic(transformed_data),
            shuffle_buffer_length=shuffle_buffer_length,
        ),
        is_train=True,
    )

    return IterableSlice(
        iter(
            DataLoader(
                IterableDataset(training_instances),
                batch_size=batch_size,
                **kwargs,
            )
        ),
        num_batches_per_epoch,
    )


def create_validation_data_loader(
    freq,
    config,
    data,
    batch_size,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        FieldName.FEAT_STATIC_CAT,
        FieldName.FEAT_STATIC_REAL,
        "past_" + FieldName.FEAT_TIME,
        "past_" + FieldName.TARGET,
        "past_" + FieldName.OBSERVED_VALUES,
        "future_" + FieldName.FEAT_TIME,
    ]

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_" + FieldName.TARGET,
        "future_" + FieldName.OBSERVED_VALUES,
    ]
    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)

    instance_splitter = create_instance_splitter(config, "validation") + SelectFields(TRAINING_INPUT_NAMES)
    validation_instances = instance_splitter.apply(transformed_data, is_train=True)

    return DataLoader(
        IterableDataset(validation_instances),
        batch_size=batch_size,
        **kwargs,
    )


def create_test_data_loader(
    freq,
    config,
    data,
    batch_size,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        FieldName.FEAT_STATIC_CAT,
        FieldName.FEAT_STATIC_REAL,
        "past_" + FieldName.FEAT_TIME,
        "past_" + FieldName.TARGET,
        "past_" + FieldName.OBSERVED_VALUES,
        "future_" + FieldName.FEAT_TIME,
    ]
    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)
    instance_splitter = create_instance_splitter(config, "test") + SelectFields(PREDICTION_INPUT_NAMES)
    test_instances = instance_splitter.apply(transformed_data, is_tran=False)

    return DataLoader(
        IterableDataset(test_instances),
        batch_size=batch_size,
        **kwargs,
    )
