..
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

HfDeepSpeedConfig
-----------------------------------------------------------------------------------------------------------------------

The :class:`~transformers.integrations.HfDeepSpeedConfig` is used to integrate Deepspeed into the 🤗 Transformer core
functionality, when :class:`~transformers.Trainer` is not used.

When using :class:`~transformers.Trainer` everything is automatically taken care of.

When not using :class:`~transformers.Trainer`, to efficiently deploy DeepSpeed stage 3, you must instantiate the
:class:`~transformers.integrations.HfDeepSpeedConfig` object before instantiating the model.

For example for a pretrained model:

.. code-block:: python

    from transformers.integrations import HfDeepSpeedConfig
    from transformers import AugoModel

    ds_config = { ... } # deepspeed config object or path to the file
    # must run before instantiating the model
    dschf = HfDeepSpeedConfig(ds_config) # keep this object alive
    model = AutoModel.from_pretrained("gpt2")
    engine = deepspeed.initialize(model=model, config_params=ds_config, ...)

or for non-pretrained model:

.. code-block:: python

    from transformers.integrations import HfDeepSpeedConfig
    from transformers import AugoModel, AutoConfig

    ds_config = { ... } # deepspeed config object or path to the file
    # must run before instantiating the model
    dschf = HfDeepSpeedConfig(ds_config) # keep this object alive
    config = AutoConfig.from_pretrained("gpt2")
    model = AutoModel.from_config(config)
    engine = deepspeed.initialize(model=model, config_params=ds_config, ...)


HfDeepSpeedConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.integrations.HfDeepSpeedConfig
    :members:
