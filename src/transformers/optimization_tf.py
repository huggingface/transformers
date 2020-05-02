# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Functions and classes related to optimization (weight updates)."""


import re

import tensorflow as tf


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(
        self, initial_learning_rate, decay_schedule_fn, warmup_steps, power=1.0, name=None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


def create_optimizer(init_lr, num_train_steps, num_warmup_steps, end_lr=0.0, optimizer_type="adamw"):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr, decay_steps=num_train_steps, end_learning_rate=end_lr,
    )
    if num_warmup_steps:
        lr_schedule = WarmUp(
            initial_learning_rate=init_lr, decay_schedule_fn=lr_schedule, warmup_steps=num_warmup_steps,
        )

    optimizer = AdamWeightDecay(
        learning_rate=lr_schedule,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["layer_norm", "bias"],
    )

    return optimizer


class AdamWeightDecay(tf.keras.optimizers.Adam):
    """Adam enables L2 weight decay and clip_by_global_norm on gradients.
  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.
  Instead we want ot decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD.
  """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay_rate=0.0,
        include_in_weight_decay=None,
        exclude_from_weight_decay=None,
        name="AdamWeightDecay",
        **kwargs
    ):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its config with WarmUp custom object."""
        custom_objects = {"WarmUp": WarmUp}
        return super(AdamWeightDecay, cls).from_config(config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["weight_decay_rate"] = tf.constant(
            self.weight_decay_rate, name="adam_weight_decay_rate"
        )

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var * apply_state[(var.device, var.dtype.base_dtype)]["weight_decay_rate"],
                use_locking=self._use_locking,
            )
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None):
        grads, tvars = list(zip(*grads_and_vars))
        return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars), name=name,)

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients["lr_t"], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({"weight_decay_rate": self.weight_decay_rate})
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


# Extracted from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py
class GradientAccumulator(object):
    """Gradient accumulation utility.
  When used with a distribution strategy, the accumulator should be called in a
  replica context. Gradients will be accumulated locally on each replica and
  without synchronization. Users should then call ``.gradients``, scale the
  gradients if required, and pass the result to ``apply_gradients``.
  """

    # We use the ON_READ synchronization policy so that no synchronization is
    # performed on assignment. To get the value, we call .value() which returns the
    # value on the current replica without synchronization.

    def __init__(self):
        """Initializes the accumulator."""
        self._gradients = []
        self._accum_steps = None

    @property
    def step(self):
        """Number of accumulated steps."""
        if self._accum_steps is None:
            self._accum_steps = tf.Variable(
                tf.constant(0, dtype=tf.int64), trainable=False, synchronization=tf.VariableSynchronization.ON_READ,
            )

        return self._accum_steps.value()

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            raise ValueError("The accumulator should be called first to initialize the gradients")
        return list(gradient.value() for gradient in self._gradients)

    def __call__(self, gradients):
        """Accumulates :obj:`gradients` on the current replica."""
        if not self._gradients:
            _ = self.step  # Create the step variable.
            self._gradients.extend(
                [
                    tf.Variable(
                        tf.zeros_like(gradient), trainable=False, synchronization=tf.VariableSynchronization.ON_READ,
                    )
                    for gradient in gradients
                ]
            )
        if len(gradients) != len(self._gradients):
            raise ValueError("Expected %s gradients, but got %d" % (len(self._gradients), len(gradients)))

        for accum_gradient, gradient in zip(self._gradients, gradients):
            accum_gradient.assign_add(gradient)

        self._accum_steps.assign_add(1)

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        if not self._gradients:
            return
        self._accum_steps.assign(0)
        for gradient in self._gradients:
            gradient.assign(tf.zeros_like(gradient))
