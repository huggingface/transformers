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
from copy import copy
from utils import parse_bert_args
from bert_data import get_generated_datum
from bert_model import PipelinedBertWithLoss
from bert_ipu import get_options
from bert import get_optimizer


@pytest.mark.ipus(4)
@pytest.mark.category1
def test_weight_decay_disabled_biases_and_norms():
    """
    Test that weight decay is successfully disabled for
    bias and norm layers.
    """
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Let's have two models:
    # - One with weight decay
    # - Another without weight decay
    # Run both for 1 training step
    # If weight decay is successfully disabled on the bias and norm
    # layers then after 1 training step these layers will be the same,
    # while the other layers will differ.

    args = """
    --config unit_test
    --lr-warmup 0.25
    --lr-schedule linear
    --batches-per-step 1
    """.split()
    config1 = transformers.BertConfig(**(vars(parse_bert_args(args))))
    config2 = copy(config1)
    opts = get_options(config1)
    opts._Popart.set("enableStochasticRounding", False)

    # Model 1 has weight decay
    config1.weight_decay = 1.0
    model1 = PipelinedBertWithLoss(config1).half().train()
    optimizer1 = get_optimizer(config1, model1)
    poptorch_model1 = poptorch.trainingModel(
        model1, opts, optimizer=optimizer1)

    # Model 2 has the same initial weights as Model 1,
    # but no weight decay
    config2.weight_decay = 0.0
    model2 = PipelinedBertWithLoss(config2).half().train()
    model2.load_state_dict(model1.state_dict())
    optimizer2 = get_optimizer(config2, model2)
    poptorch_model2 = poptorch.trainingModel(
        model2, opts, optimizer=optimizer2)

    # Run 1 training step on both with mock data
    def mock_data():
        return get_generated_datum(config1)

    datum = mock_data()
    output1 = poptorch_model1(*datum)
    output2 = poptorch_model2(*datum)

    # Go through the state_dicts comparing the weights
    # if the name is like LayerNorm or bias then they should be
    # the same. Otherwise they should be different.
    for name, tensor1 in model1.state_dict().items():
        tensor2 = model2.state_dict()[name]

        if tensor1.dtype is not torch.int64:
            if ("bias" in name) or ("LayerNorm" in name):
                assert torch.allclose(tensor1, tensor2)
            else:
                assert not torch.allclose(tensor1, tensor2)
