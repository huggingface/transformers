# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import poptorch
import pytest
import numpy as np
from transformers import BertTokenizer, BertConfig
from bert_model import PipelinedBertWithLoss
from bert_ipu import get_options
from utils import parse_bert_args


@pytest.mark.ipus(4)
@pytest.mark.category1
@pytest.mark.parametrize("embedding_serialization, recompute_checkpoint", [(1, False), (5, True)])
def test_ipu_cpu_match(recompute_checkpoint, embedding_serialization):
    """
    Test that the BERT model ran on IPU approximately matches that same
    model ran on the CPU.
    """
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Config
    args = """
    --config unit_test
    --lr-schedule constant
    --layers-per-ipu 0 3
    --vocab-size 30400
    --batch-size 10
    --batches-per-step 1
    --gradient-accumulation 10
    --enable-half-partials False
    --optimizer AdamW
    --learning-rate 0.001
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    config.recompute_checkpoint_every_layer = recompute_checkpoint
    config.embedding_serialization = embedding_serialization

    # Models and options
    opts = get_options(config)
    opts.anchorMode(poptorch.AnchorMode.Final)
    model_cpu = PipelinedBertWithLoss(config).train()
    model_ipu = PipelinedBertWithLoss(config).train()
    model_ipu.load_state_dict(model_cpu.state_dict())

    # Check that copy was successful
    assert model_ipu is not model_cpu
    assert all([(a == b).all() for a, b in zip(
        model_cpu.parameters(), model_ipu.parameters())]) is True

    optimizer_cpu = torch.optim.AdamW(model_cpu.parameters(), lr=0.001)
    optimizer_ipu = poptorch.optim.AdamW(model_ipu.parameters(), lr=0.001, loss_scaling=1.0)
    poptorch_model = poptorch.trainingModel(model_ipu, opts, optimizer=optimizer_ipu)

    # Input
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer("Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute yo"
                       "Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute yo"
                       "Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute yo"
                       "Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute", return_tensors="pt")
    inputs['labels'] = torch.randint(0, config.vocab_size, [1, config.mask_tokens], dtype=torch.long)
    inputs['next_sentence_label'] = torch.randint(0, 1, [1], dtype=torch.long)
    inputs['masked_lm_positions'] = torch.randint(0, config.sequence_length, [1, config.mask_tokens], dtype=torch.long)

    batch_size = config.batch_size

    batch = (inputs['input_ids'].repeat(batch_size, 1),
             inputs['attention_mask'].repeat(batch_size, 1),
             inputs['token_type_ids'].repeat(batch_size, 1),
             inputs['masked_lm_positions'].repeat(batch_size, 1),
             inputs['labels'].repeat(batch_size, 1),
             inputs['next_sentence_label'].repeat(batch_size, 1))

    batch_cpu = (inputs['input_ids'].repeat(1, 1),
                 inputs['attention_mask'].repeat(1, 1),
                 inputs['token_type_ids'].repeat(1, 1),
                 inputs['masked_lm_positions'].repeat(1, 1),
                 inputs['labels'].repeat(1, 1),
                 inputs['next_sentence_label'].repeat(1, 1))

    # Training Loop
    for step in range(10):
        # Step CPU model
        optimizer_cpu.zero_grad()
        for b in range(batch_size):
            cpu_output = model_cpu(*batch_cpu)
            cpu_loss = cpu_output[0]
            cpu_loss.div(batch_size).backward()
        optimizer_cpu.step()

        # Step IPU Model
        ipu_output = poptorch_model(*batch)
        ipu_loss = ipu_output[0]

        with torch.no_grad():
            print(f"CPU Loss: {cpu_loss}, IPU Loss: {ipu_loss}")
            # Check the losses are approximately equal
            assert np.allclose(cpu_loss.numpy(), ipu_loss.numpy(), atol=1e-6)
