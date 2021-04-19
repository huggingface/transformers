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

import pytest
from transformers import BertConfig
from bert_model import _get_layer_ipu
from utils import parse_bert_args


@pytest.mark.category1
def test_single_value_layers_per_ipu():
    args = """
    --config unit_test
    --layers-per-ipu 1
    --num-hidden-layers 4
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    assert config.layers_per_ipu == [1, 1, 1, 1]


@pytest.mark.category1
def test_multi_value_layers_per_ipu():
    args = """
    --config unit_test
    --layers-per-ipu 1 2 3 4
    --num-hidden-layers 10
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    assert config.layers_per_ipu == [1, 2, 3, 4]

    args = """
    --config unit_test
    --layers-per-ipu 0 3 3 4
    --num-hidden-layers 10
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    assert config.layers_per_ipu == [0, 3, 3, 4]


@pytest.mark.category1
def test_invalid_layers_per_ipu():
    args = """
    --config unit_test
    --layers-per-ipu 1 1 1 1
    --num-hidden-layers 3
    """.split()
    with pytest.raises(SystemExit):
        config = BertConfig(**(vars(parse_bert_args(args))))

    args = """
    --config unit_test
    --layers-per-ipu 4
    --num-hidden-layers 3
    """.split()
    with pytest.raises(SystemExit):
        config = BertConfig(**(vars(parse_bert_args(args))))

    args = """
    --config unit_test
    --layers-per-ipu 0 1 2 1
    --num-hidden-layers 3
    """.split()
    with pytest.raises(SystemExit):
        config = BertConfig(**(vars(parse_bert_args(args))))

    args = """
    --config unit_test
    --layers-per-ipu 0 1 1 1 1
    --num-hidden-layers 3
    """.split()
    with pytest.raises(SystemExit):
        config = BertConfig(**(vars(parse_bert_args(args))))


@pytest.mark.category1
def test_single_value_matmul_prop():
    # Matmul proportion on all IPUs, not just encoder IPUs
    args = """
    --config unit_test
    --layers-per-ipu 1
    --num-hidden-layers 4
    --matmul-proportion 0.2
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    assert config.matmul_proportion == [0.2, 0.2, 0.2, 0.2]


@pytest.mark.category1
def test_multi_value_matmul_prop():
    args = """
    --config unit_test
    --layers-per-ipu 3 7 7 7
    --num-hidden-layers 24
    --matmul-proportion 0.15 0.3 0.3 0.3
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    assert config.matmul_proportion == [0.15, 0.3, 0.3, 0.3]

    # Invalid inputs
    args = """
    --config unit_test
    --layers-per-ipu 3 7 7 7
    --num-hidden-layers 24
    --matmul-proportion 0.15 0.3 0.3
    """.split()
    with pytest.raises(SystemExit):
        config = BertConfig(**(vars(parse_bert_args(args))))

    args = """
    --config unit_test
    --layers-per-ipu 3 7 7 7
    --num-hidden-layers 24
    --matmul-proportion 0.15 0.3 0.3 0.3 0.3
    """.split()
    with pytest.raises(SystemExit):
        config = BertConfig(**(vars(parse_bert_args(args))))


@pytest.mark.category1
def test_get_layer_ipu():
    args = """
    --config unit_test
    --layers-per-ipu 2
    --num-hidden-layers 12
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    assert (_get_layer_ipu(config.layers_per_ipu) ==
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
            )

    args = """
    --config unit_test
    --layers-per-ipu 2 2 2 2 2 1
    --num-hidden-layers 11
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    assert (_get_layer_ipu(config.layers_per_ipu) ==
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
            )

    args = """
    --config unit_test
    --layers-per-ipu 0 1 1 1
    --num-hidden-layers 3
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    assert (_get_layer_ipu(config.layers_per_ipu) ==
            [1, 2, 3]
            )
