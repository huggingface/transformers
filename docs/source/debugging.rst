..
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.



Debugging
=========

Activations Overflow
--------------------

.. note::

   This feature is currently available for PyTorch-only.

If you start getting ``loss=NaN`` or the model inhibits some other abnormal behavior due to ``inf``s or ``nan``s one needs to discover where the first overflow happens and what led to it. Luckily you can accomplish that easily by activating a special module that will do the detection automatically.

If you're using :class:`~transformers.Trainer`, you just need to add:

.. code-block:: bash

   --debug activation_overflow

to the normal command line arguments, or pass ``debug="activation_overflow"`` when creating the :class:`~transformers.Trainer` object.

If you're using your own trainer you can just do:

.. code-block:: python

   from .debug_utils import DebugActivationOverflow
   debug_overflow = DebugActivationOverflow(model)

``DebugActivationOverflow`` inserts hooks into the model that will test each input and output and as soon as ``inf`` or ``nan`` is detected in at least one element, the program will assert and print a report like this:

.. code-block::

   < [0] encoder.block.2.layer.1.DenseReluDense.wo: Linear: output has infs


   last 40 frames:
   abs_max= 5.96e+02 < [0] encoder.block.1.layer.1.DenseReluDense.dropout: Dropout: output
   abs_max= 5.96e+02 > [0] encoder.block.1.layer.1.DenseReluDense.wo: Linear: input[0]
   abs_max= 3.17e+03 < [0] encoder.block.1.layer.1.DenseReluDense.wo: Linear: output
   abs_max= 2.57e+00 > [0] encoder.block.1.layer.1.DenseReluDense: T5DenseGatedGeluDense: input[0]
   abs_max= 3.17e+03 < [0] encoder.block.1.layer.1.DenseReluDense: T5DenseGatedGeluDense: output
   abs_max= 3.17e+03 > [0] encoder.block.1.layer.1.dropout: Dropout: input[0]
   abs_max= 3.52e+03 < [0] encoder.block.1.layer.1.dropout: Dropout: output
   abs_max= 1.58e+03 > [0] encoder.block.1.layer.1: T5LayerFF: input[0]
   abs_max= 4.04e+03 < [0] encoder.block.1.layer.1: T5LayerFF: output
   abs_max= 1.51e+03 > [0] encoder.block.1: T5Block: input[0]
   abs_max= 4.04e+03 < [0] encoder.block.1: T5Block: output[0]
   abs_max= 1.00e+04 < [0] encoder.block.1: T5Block: output[2]
   abs_max= 4.04e+03 > [0] encoder.block.2.layer.0.layer_norm: T5LayerNorm: input[0]
   abs_max= 2.69e+00 < [0] encoder.block.2.layer.0.layer_norm: T5LayerNorm: output
   abs_max= 2.69e+00 > [0] encoder.block.2.layer.0.SelfAttention.q: Linear: input[0]
   abs_max= 1.13e+00 < [0] encoder.block.2.layer.0.SelfAttention.q: Linear: output
   abs_max= 2.69e+00 > [0] encoder.block.2.layer.0.SelfAttention.k: Linear: input[0]
   abs_max= 1.69e+01 < [0] encoder.block.2.layer.0.SelfAttention.k: Linear: output
   abs_max= 2.69e+00 > [0] encoder.block.2.layer.0.SelfAttention.v: Linear: input[0]
   abs_max= 8.92e+00 < [0] encoder.block.2.layer.0.SelfAttention.v: Linear: output
   abs_max= 7.59e+00 > [0] encoder.block.2.layer.0.SelfAttention.o: Linear: input[0]
   abs_max= 2.83e+02 < [0] encoder.block.2.layer.0.SelfAttention.o: Linear: output
   abs_max= 2.69e+00 > [0] encoder.block.2.layer.0.SelfAttention: T5Attention: input[0]
   abs_max= 2.83e+02 < [0] encoder.block.2.layer.0.SelfAttention: T5Attention: output[0]
   abs_max= 1.00e+04 < [0] encoder.block.2.layer.0.SelfAttention: T5Attention: output[2]
   abs_max= 2.83e+02 > [0] encoder.block.2.layer.0.dropout: Dropout: input[0]
   abs_max= 3.14e+02 < [0] encoder.block.2.layer.0.dropout: Dropout: output
   abs_max= 4.04e+03 > [0] encoder.block.2.layer.0: T5LayerSelfAttention: input[0]
   abs_max= 4.06e+03 < [0] encoder.block.2.layer.0: T5LayerSelfAttention: output[0]
   abs_max= 1.00e+04 < [0] encoder.block.2.layer.0: T5LayerSelfAttention: output[2]
   abs_max= 4.06e+03 > [0] encoder.block.2.layer.1.layer_norm: T5LayerNorm: input[0]
   abs_max= 6.00e+00 < [0] encoder.block.2.layer.1.layer_norm: T5LayerNorm: output
   abs_max= 6.00e+00 > [0] encoder.block.2.layer.1.DenseReluDense.wi_0: Linear: input[0]
   abs_max= 5.18e+01 < [0] encoder.block.2.layer.1.DenseReluDense.wi_0: Linear: output
   abs_max= 6.00e+00 > [0] encoder.block.2.layer.1.DenseReluDense.wi_1: Linear: input[0]
   abs_max= 3.14e+02 < [0] encoder.block.2.layer.1.DenseReluDense.wi_1: Linear: output
   abs_max= 1.62e+04 > [0] encoder.block.2.layer.1.DenseReluDense.dropout: Dropout: input[0]
   abs_max= 1.80e+04 < [0] encoder.block.2.layer.1.DenseReluDense.dropout: Dropout: output
   abs_max= 1.80e+04 > [0] encoder.block.2.layer.1.DenseReluDense.wo: Linear: input[0]
   abs_max=      inf < [0] encoder.block.2.layer.1.DenseReluDense.wo: Linear: output

The left column shows the value of the absolute largest element, so if you have a closer look the last few frames, the inputs and outputs were in the range of 10000. So when this training was done under mixed precision the very last step overflowed (since under ``fp16`` the largest number before ``inf`` is ``64e3``). To avoid overflows under ``fp16`` the activations must remain way below ``1e4``, because ``1e4*1e4 = 1e8`` so any matrix multiply with large activations is going to lead to overflow.

The trace then prints the batch number (here ``[0]`` means the problem occurred on the first batch).

Then comes the fully qualified entry from the ``state_dict``, e.g.: ``encoder.block.2.layer.0.layer_norm``, so you can easily see where the problem happens and what was happening just before it.

The second to last entry show the name of the class the ``forward`` belongs to, and whether the report is for an input or an output and its index if either is a tuple. Only tensor variables are reported.

Another shortcut in the first columns ``>`` is for input variable, ``<`` is for output.

Let's look at:

.. code-block::

   abs_max= 1.62e+04 > [0] encoder.block.2.layer.1.DenseReluDense.dropout: Dropout: input[0]
   abs_max= 1.80e+04 < [0] encoder.block.2.layer.1.DenseReluDense.dropout: Dropout: output

This is a report for ``Dropout.forward`` function with the first entry for the only input and the second for the only output. You can see that it was called from an attribute ``dropout`` inside ``DenseReluDense`` class. We can see that it happened during the first layer, of the 2nd block, during the very first batch. Finally the absolute largest input elements was ``1.62e+04`` and same for the output was ``1.80e+04``.

Going back to the full report, to act on it and to fix the problem, we need to go a few frames up where the numbers started to go up and most likely switch to the ``fp32`` mode here, so that the numbers don't overflow when multiplied or summed up. Of course, there might be other solutions.

Since the automatic detector only reports inputs and outputs, once you know where to look, you may want to analyse the intermediary stages of ``forward`` as well. In such a case you can use the helper function to inject the detector where you want it, for example:

.. code-block::

   from debug_utils import detect_overflow

   class T5LayerFF(nn.Module):
       [...]
       def forward(self, hidden_states):
           forwarded_states = self.layer_norm(hidden_states)
           detect_overflow(forwarded_states, "after layer_norm")
           forwarded_states = self.DenseReluDense(forwarded_states)
           detect_overflow(forwarded_states, "after DenseReluDense")
           return hidden_states + self.dropout(forwarded_states)

You can see that we added 2 of these and now we can know the absolute largest numbers for ``forwarded_states`` at 2 different stages.
