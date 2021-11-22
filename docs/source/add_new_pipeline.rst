.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

How to add a pipeline to 🤗 Transformers?
=======================================================================================================================

First and foremost, you need to decide the raw entries the pipeline will be able to take. It can be strings, raw bytes,
dictionaries or whatever seems to be the most likely desired input. Try to keep these inputs as pure Python as possible
as it makes compatibility easier (even through other languages via JSON). Those will be the :obj:`inputs` of the
pipeline (:obj:`preprocess`).

Then define the :obj:`outputs`. Same policy as the :obj:`inputs`. The simpler, the better. Those will be the outputs of
:obj:`postprocess` method.

Start by inheriting the base class :obj:`Pipeline`. with the 4 methods needed to implement :obj:`preprocess`,
:obj:`_forward`, :obj:`postprocess` and :obj:`_sanitize_parameters`.


.. code-block::

    from transformers import Pipeline

    class MyPipeline(Pipeline):
        def _sanitize_parameters(self, **kwargs):
            preprocess_kwargs = {}
            if "maybe_arg" in kwargs:
                preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
            return preprocess_kwargs, {}, {}

        def preprocess(self, inputs, maybe_arg=2):
            model_input = Tensor(....)
            return {"model_input": model_input}

        def _forward(self, model_inputs):
            # model_inputs == {"model_input": model_input}
            outputs = self.model(**model_inputs)
            # Maybe {"logits": Tensor(...)}
            return outputs

        def postprocess(self, model_outputs):
            best_class = model_outputs["logits"].softmax(-1)
            return best_class


The structure of this breakdown is to support relatively seamless support for CPU/GPU, while supporting doing
pre/postprocessing on the CPU on different threads

:obj:`preprocess` will take the originally defined inputs, and turn them into something feedable to the model. It might
contain more information and is usually a :obj:`Dict`.

:obj:`_forward` is the implementation detail and is not meant to be called directly. :obj:`forward` is the preferred
called method as it contains safeguards to make sure everything is working on the expected device. If anything is
linked to a real model it belongs in the :obj:`_forward` method, anything else is in the preprocess/postprocess.

:obj:`postprocess` methods will take the output of :obj:`_forward` and turn it into the final output that were decided
earlier.

:obj:`_sanitize_parameters` exists to allow users to pass any parameters whenever they wish, be it at initialization
time ``pipeline(...., maybe_arg=4)`` or at call time ``pipe = pipeline(...); output = pipe(...., maybe_arg=4)``.

The returns of :obj:`_sanitize_parameters` are the 3 dicts of kwargs that will be passed directly to :obj:`preprocess`,
:obj:`_forward` and :obj:`postprocess`. Don't fill anything if the caller didn't call with any extra parameter. That
allows to keep the default arguments in the function definition which is always more "natural".

A classic example would be a :obj:`top_k` argument in the post processing in classification tasks.

.. code-block::

    >>> pipe = pipeline("my-new-task")
    >>> pipe("This is a test")
    [{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}
    {"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]

    >>> pipe("This is a test", top_k=2)
    [{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]

In order to achieve that, we'll update our :obj:`postprocess` method with a default parameter to :obj:`5`. and edit
:obj:`_sanitize_parameters` to allow this new parameter.


.. code-block::


        def postprocess(self, model_outputs, top_k=5):
            best_class = model_outputs["logits"].softmax(-1)
            # Add logic to handle top_k
            return best_class

        def _sanitize_parameters(self, **kwargs):
            preprocess_kwargs = {}
            if "maybe_arg" in kwargs:
                preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

            postprocess_kwargs = {}
            if "top_k" in kwargs:
                preprocess_kwargs["top_k"] = kwargs["top_k"]
            return preprocess_kwargs, {}, postprocess_kwargs

Try to keep the inputs/outputs very simple and ideally JSON-serializable as it makes the pipeline usage very easy
without requiring users to understand new kind of objects. It's also relatively common to support many different types
of arguments for ease of use (audio files, can be filenames, URLs or pure bytes)



Adding it to the list of supported tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Go to ``src/transformers/pipelines/__init__.py`` and fill in :obj:`SUPPORTED_TASKS` with your newly created pipeline.
If possible it should provide a default model.

Adding tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new file ``tests/test_pipelines_MY_PIPELINE.py`` with example with the other tests.

The :obj:`run_pipeline_test` function will be very generic and run on small random models on every possible
architecture as defined by :obj:`model_mapping` and :obj:`tf_model_mapping`.

This is very important to test future compatibility, meaning if someone adds a new model for
:obj:`XXXForQuestionAnswering` then the pipeline test will attempt to run on it. Because the models are random it's
impossible to check for actual values, that's why There is a helper :obj:`ANY` that will simply attempt to match the
output of the pipeline TYPE.

You also *need* to implement 2 (ideally 4) tests.

- :obj:`test_small_model_pt` : Define 1 small model for this pipeline (doesn't matter if the results don't make sense)
  and test the pipeline outputs. The results should be the same as :obj:`test_small_model_tf`.
- :obj:`test_small_model_tf` : Define 1 small model for this pipeline (doesn't matter if the results don't make sense)
  and test the pipeline outputs. The results should be the same as :obj:`test_small_model_pt`.
- :obj:`test_large_model_pt` (:obj:`optional`): Tests the pipeline on a real pipeline where the results are supposed to
  make sense. These tests are slow and should be marked as such. Here the goal is to showcase the pipeline and to make
  sure there is no drift in future releases
- :obj:`test_large_model_tf` (:obj:`optional`): Tests the pipeline on a real pipeline where the results are supposed to
  make sense. These tests are slow and should be marked as such. Here the goal is to showcase the pipeline and to make
  sure there is no drift in future releases
