import unittest

from transformers import is_tf_available

from .utils import require_tf


if is_tf_available():
    import tensorflow as tf
    from tensorflow.python.eager import context
    from tensorflow.python.framework import ops
    from transformers import create_optimizer, GradientAccumulator


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
        physical_devices = tf.config.experimental.list_physical_devices("CPU")
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(), tf.config.experimental.VirtualDeviceConfiguration()],
        )

        devices = tf.config.experimental.list_logical_devices(device_type="CPU")
        strategy = tf.distribute.MirroredStrategy(devices=[device.name for device in devices])

        with strategy.scope():
            accumulator = GradientAccumulator()
            variable = tf.Variable([4.0, 3.0])
            optimizer = create_optimizer(5e-5, 10, 5)
            gradient_placeholder = tf.Variable([0.0, 0.0], trainable=False)

        def accumulate_on_replica(gradient):
            accumulator([gradient])

        def apply_on_replica():
            optimizer.apply_gradients(list(zip(accumulator.gradients, [variable])), 1.0)

        @tf.function
        def accumulate(grad1, grad2):
            with strategy.scope():
                gradient_placeholder.values[0].assign(grad1)
                gradient_placeholder.values[1].assign(grad2)
                strategy.experimental_run_v2(accumulate_on_replica, args=(gradient_placeholder,))

        @tf.function
        def apply_grad():
            with strategy.scope():
                strategy.experimental_run_v2(apply_on_replica)

        accumulate([1.0, 2.0], [-1.0, 1.0])
        accumulate([3.0, -1.0], [-1.0, -1.0])
        accumulate([-2.0, 2.0], [3.0, -2.0])
        self.assertEqual(accumulator.step, 3)
        self.assertListAlmostEqual(accumulator._gradients[0].values[0].value().numpy().tolist(), [2.0, 3.0], tol=1e-2)
        self.assertListAlmostEqual(accumulator._gradients[0].values[1].value().numpy().tolist(), [1.0, -2.0], tol=1e-2)
        apply_grad()
        self.assertListAlmostEqual(variable.value().numpy().tolist(), [4.0, 3.0], tol=1e-2)
        accumulator.reset()
        self.assertEqual(accumulator.step, 0)
        self.assertListAlmostEqual(accumulator._gradients[0].values[0].value().numpy().tolist(), [0.0, 0.0], tol=1e-2)
        self.assertListAlmostEqual(accumulator._gradients[0].values[1].value().numpy().tolist(), [0.0, 0.0], tol=1e-2)
