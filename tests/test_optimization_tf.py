# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest

from packaging import version

from transformers import is_tf_available
from transformers.testing_utils import require_tf


if is_tf_available():
    import tensorflow as tf
    from tensorflow.python.eager import context
    from tensorflow.python.framework import ops

    from transformers import GradientAccumulator, create_optimizer


@require_tf
class OptimizationFTest(unittest.TestCase):
    def assertListAlmostEqual(self, list1, list2, tol):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, delta=tol)

    def testGradientAccumulator(self):
        accumulator = GradientAccumulator()
        accumulator([tf.constant([1.0, 2.0])])
        accumulator([tf.constant([-2.0, 1.0])])
        accumulator([tf.constant([-1.0, 2.0])])
        with self.assertRaises(ValueError):
            accumulator([tf.constant([1.0, 1.0]), tf.constant([2.0, 2.0])])
        self.assertEqual(accumulator.step, 3)
        self.assertEqual(len(accumulator.gradients), 1)
        self.assertListAlmostEqual(accumulator.gradients[0].numpy().tolist(), [-2.0, 5.0], tol=1e-2)
        accumulator.reset()
        self.assertEqual(accumulator.step, 0)
        self.assertListAlmostEqual(accumulator.gradients[0].numpy().tolist(), [0.0, 0.0], tol=1e-2)

    def testGradientAccumulatorDistributionStrategy(self):
        context._context = None
        ops.enable_eager_execution_internal()
        physical_devices = tf.config.list_physical_devices("CPU")
        if len(physical_devices) == 1:
            tf.config.set_logical_device_configuration(
                physical_devices[0], [tf.config.LogicalDeviceConfiguration(), tf.config.LogicalDeviceConfiguration()]
            )
        devices = tf.config.list_logical_devices(device_type="CPU")
        strategy = tf.distribute.MirroredStrategy(devices=devices[:2])

        with strategy.scope():
            accumulator = GradientAccumulator()
            variable = tf.Variable([4.0, 3.0])
            optimizer, _ = create_optimizer(5e-5, 10, 5)
            gradient_placeholder = tf.Variable([0.0, 0.0], trainable=False)

        def accumulate_on_replica(gradient):
            accumulator([gradient])

        def apply_on_replica():
            optimizer.apply_gradients(list(zip(accumulator.gradients, [variable])))

        @tf.function
        def accumulate(grad1, grad2):
            with strategy.scope():
                local_variables = strategy.experimental_local_results(gradient_placeholder)
                local_variables[0].assign(grad1)
                local_variables[1].assign(grad2)
                if version.parse(tf.version.VERSION) >= version.parse("2.2"):
                    strategy.run(accumulate_on_replica, args=(gradient_placeholder,))
                else:
                    strategy.experimental_run_v2(accumulate_on_replica, args=(gradient_placeholder,))

        @tf.function
        def apply_grad():
            with strategy.scope():
                if version.parse(tf.version.VERSION) >= version.parse("2.2"):
                    strategy.run(apply_on_replica)
                else:
                    strategy.experimental_run_v2(apply_on_replica)

        def _check_local_values(grad1, grad2):
            values = strategy.experimental_local_results(accumulator._gradients[0])
            self.assertListAlmostEqual(values[0].value(), grad1, tol=1e-2)
            self.assertListAlmostEqual(values[1].value(), grad2, tol=1e-2)

        accumulate([1.0, 2.0], [-1.0, 1.0])
        accumulate([3.0, -1.0], [-1.0, -1.0])
        accumulate([-2.0, 2.0], [3.0, -2.0])
        self.assertEqual(accumulator.step, 3)
        _check_local_values([2.0, 3.0], [1.0, -2.0])
        apply_grad()
        self.assertListAlmostEqual(variable.value(), [4.0, 3.0], tol=1e-2)
        accumulator.reset()
        self.assertEqual(accumulator.step, 0)
        _check_local_values([0.0, 0.0], [0.0, 0.0])
