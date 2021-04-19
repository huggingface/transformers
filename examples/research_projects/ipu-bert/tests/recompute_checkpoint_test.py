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
import pytest
import json
from transformers import BertConfig
from bert_model import PipelinedBertWithLoss
from utils import parse_bert_args
from bert_ipu import get_options
from bert_optimization import get_optimizer
from bert_data import get_generated_datum


@pytest.mark.category(1)
@pytest.mark.ipus(4)
def test_checkpoint_in_ir():
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Config
    args = """
    --config unit_test
    --lr-schedule constant
    --layers-per-ipu 0 3
    --vocab-size 30400
    --weight-decay 0.0
    --recompute-checkpoint-every-layer True
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))

    assert config.recompute_checkpoint_every_layer is True

    # Execution parameters
    opts = get_options(config)
    model = PipelinedBertWithLoss(config).half().train()
    optimizer = get_optimizer(config, model)
    poptorch_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    # Compile model
    datum = get_generated_datum(config)
    poptorch_model.compile(*datum)
    ir = json.loads(poptorch_model._debugGetPopartIR())

    assert any(["Checkpoint" in node["name"] for node in ir["maingraph"]
                ]), ("Popart IR should contain a checkpoint")
    # Stash: 5 inputs + 3 for the transformer layers
    exp_num_stash = 5+3
    print(sum(["Stash" in node["type"] for node in ir["maingraph"]]))
    assert sum([
        "Stash" in node["type"] for node in ir["maingraph"]
    ]) == exp_num_stash, ("Both the graph input and the checkpoint(s) "
                          "should be stashed")


@pytest.mark.category(1)
@pytest.mark.ipus(4)
def test_checkpoint_not_in_ir():
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Config
    args = """
    --config unit_test
    --lr-schedule constant
    --layers-per-ipu 0 3
    --vocab-size 30400
    --weight-decay 0.0
    --recompute-checkpoint-every-layer False
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))

    assert config.recompute_checkpoint_every_layer is False

    # Execution parameters
    opts = get_options(config)
    model = PipelinedBertWithLoss(config).half().train()
    optimizer = get_optimizer(config, model)
    poptorch_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    # Compile model
    datum = get_generated_datum(config)
    poptorch_model.compile(*datum)
    ir = json.loads(poptorch_model._debugGetPopartIR())
    assert not any(["Checkpoint" in node["name"] for node in ir["maingraph"]
                    ]), ("Popart IR should contain a checkpoint")

    # Stash: 5 inputs, and 1 stash for transformers on ipu1
    exp_num_stash = 5 + 1
    assert sum([
        "Stash" in node["type"] for node in ir["maingraph"]
    ]) == exp_num_stash, ("Both the graph input and the checkpoint(s) "
                          "should be stashed")
    print(sum(["Stash" in node["type"] for node in ir["maingraph"]]))
