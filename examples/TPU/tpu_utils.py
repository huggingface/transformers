# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
""" === Under active development === Loading a TPUStrategy

Especially https://github.com/GoogleCloudPlatform/training-data-analyst/blob/tf2/courses/fast-and-lean-data-science/01_MNIST_TPU_Keras.ipynb
"""

import tensorflow as tf


def get_tpu():
    tpu = None

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    except ValueError as e:
        print(e)

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://192.168.32.2:8470")
    except ValueError as e:
        print(e)

    # Select appropriate distribution strategy
    if tpu:
        # TF 2.0 change here: experimental_connect_to_cluster and initialize_tpu_system are now necessary
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        # TF 2.0 change here: steps_per_run does not exist anymore and is not needed
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    else:
        strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
        print('Running on CPU or GPU')
    print("Number of accelerators: ", strategy.num_replicas_in_sync)

    return strategy, strategy.num_replicas_in_sync
