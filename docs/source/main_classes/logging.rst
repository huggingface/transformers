.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Logging
-----------------------------------------------------------------------------------------------------------------------

🤗 Transformers has a centralized logging system, so that you can setup the verbosity of the library easily.

Currently the default verbosity of the library is ``WARNING``.

To change the level of verbosity, just use one of the direct setters. For instance, here is how to change the verbosity
to the INFO level.

.. code-block:: python

    import transformers
    transformers.logging.set_verbosity_info()

You can also use the environment variable ``TRANSFORMERS_VERBOSITY`` to override the default verbosity. You can set it
to one of the following: ``debug``, ``info``, ``warning``, ``error``, ``critical``. For example:

.. code-block:: bash

    TRANSFORMERS_VERBOSITY=error ./myprogram.py

All the methods of this logging module are documented below, the main ones are
:func:`transformers.logging.get_verbosity` to get the current level of verbosity in the logger and
:func:`transformers.logging.set_verbosity` to set the verbosity to the level of your choice. In order (from the least
verbose to the most verbose), those levels (with their corresponding int values in parenthesis) are:

- :obj:`transformers.logging.CRITICAL` or :obj:`transformers.logging.FATAL` (int value, 50): only report the most
  critical errors.
- :obj:`transformers.logging.ERROR` (int value, 40): only report errors.
- :obj:`transformers.logging.WARNING` or :obj:`transformers.logging.WARN` (int value, 30): only reports error and
  warnings. This the default level used by the library.
- :obj:`transformers.logging.INFO` (int value, 20): reports error, warnings and basic information.
- :obj:`transformers.logging.DEBUG` (int value, 10): report all information.

Base setters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformers.logging.set_verbosity_error

.. autofunction:: transformers.logging.set_verbosity_warning

.. autofunction:: transformers.logging.set_verbosity_info

.. autofunction:: transformers.logging.set_verbosity_debug

Other functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformers.logging.get_verbosity

.. autofunction:: transformers.logging.set_verbosity

.. autofunction:: transformers.logging.get_logger

.. autofunction:: transformers.logging.enable_explicit_format

.. autofunction:: transformers.logging.reset_format
