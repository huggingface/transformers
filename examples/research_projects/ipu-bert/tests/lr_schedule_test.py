# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import torch
import poptorch
import transformers
import pytest
from utils import parse_bert_args
from bert_data import get_generated_datum
from bert_model import PipelinedBertWithLoss
from bert_optimization import get_optimizer
from bert_ipu import get_options
from bert import get_lr_scheduler


@pytest.mark.ipus(4)
@pytest.mark.category1
def test_lrschedule_changes_lr():
    """
    Test that pytorch LR scheduler is correctly changing the learning rate
    in poptorch
    """
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Get args and put in config
    args = """
    --config unit_test
    --lr-warmup 0.25
    --lr-schedule linear
    """.split()
    config = transformers.BertConfig(**(vars(parse_bert_args(args))))
    opts = get_options(config)

    # IPU Model and Optimizer
    model = PipelinedBertWithLoss(config).half().train()
    optimizer = get_optimizer(config, model)
    scheduler = get_lr_scheduler(optimizer, config.lr_schedule, config.lr_warmup, config.training_steps)
    poptorch_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    def mock_data():
        return get_generated_datum(config)

    # Compile the model
    poptorch_model.compile(*mock_data())

    # Starting lr should be 0.0
    assert poptorch_model._dict_optimizer["groups"][0]["learningRate"][0] == 0.0

    # Run for warmup+1 steps to get to peak
    warmup_steps = int(config.lr_warmup * config.training_steps)
    for _ in range(warmup_steps + 1):
        outputs = poptorch_model(*mock_data())
        scheduler.step()
        poptorch_model.setOptimizer(optimizer)

    # After warmup+1 steps LR should = 1.0
    assert poptorch_model._dict_optimizer["groups"][0]["learningRate"][0] == config.learning_rate

    # run the remaining steps
    for _ in range(warmup_steps + 1, config.training_steps):
        outputs = poptorch_model(*mock_data())
        scheduler.step()
        poptorch_model.setOptimizer(optimizer)

    # LR should have decreased from the peak
    assert poptorch_model._dict_optimizer["groups"][0]["learningRate"][0] < config.learning_rate
    assert poptorch_model._dict_optimizer["groups"][0]["learningRate"][0] > 0.0

    # Running beyond the schedule sets lr=0.0
    for _ in range(config.training_steps, config.training_steps + 1):
        outputs = poptorch_model(*mock_data())
        scheduler.step()
        poptorch_model.setOptimizer(optimizer)
    assert poptorch_model._dict_optimizer["groups"][0]["learningRate"][0] == 0.0


@pytest.mark.ipus(4)
@pytest.mark.category1
def test_constant_lrschedule():
    """
    Test that lr schedule "constant" results in unchanging LR
    """
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    args = """
    --config unit_test
    --lr-schedule constant
    """.split()
    config = transformers.BertConfig(**(vars(parse_bert_args(args))))
    opts = get_options(config)

    # IPU Model and Optimizer
    model = PipelinedBertWithLoss(config).half().train()
    optimizer = get_optimizer(config, model)
    scheduler = get_lr_scheduler(optimizer, "constant")
    poptorch_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    def mock_data():
        return get_generated_datum(config)

    # Compile the model
    poptorch_model.compile(*mock_data())

    # Starting lr should be 1.0
    assert poptorch_model._dict_optimizer["groups"][0]["learningRate"][0] == config.learning_rate

    # Run for some steps
    for _ in range(5):
        outputs = poptorch_model(*mock_data())
        scheduler.step()
        poptorch_model.setOptimizer(optimizer)

    # LR should be unchanged
    assert poptorch_model._dict_new_optimizer["groups"][0]["learningRate"][0] == config.learning_rate
