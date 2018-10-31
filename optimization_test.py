# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import optimization
import tensorflow as tf


class OptimizationTest(tf.test.TestCase):

  def test_adam(self):
    with self.test_session() as sess:
      w = tf.get_variable(
          "w",
          shape=[3],
          initializer=tf.constant_initializer([0.1, -0.2, -0.1]))
      x = tf.constant([0.4, 0.2, -0.5])
      loss = tf.reduce_mean(tf.square(x - w))
      tvars = tf.trainable_variables()
      grads = tf.gradients(loss, tvars)
      global_step = tf.train.get_or_create_global_step()
      optimizer = optimization.AdamWeightDecayOptimizer(learning_rate=0.2)
      train_op = optimizer.apply_gradients(zip(grads, tvars), global_step)
      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      sess.run(init_op)
      for _ in range(100):
        sess.run(train_op)
      w_np = sess.run(w)
      self.assertAllClose(w_np.flat, [0.4, 0.2, -0.5], rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  tf.test.main()
