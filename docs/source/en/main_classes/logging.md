<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Logging

🤗 Transformers has a centralized logging system, so that you can setup the verbosity of the library easily.

Currently the default verbosity of the library is `WARNING`.

To change the level of verbosity, just use one of the direct setters. For instance, here is how to change the verbosity
to the INFO level.

```python
import transformers

transformers.logging.set_verbosity_info()
```

You can also use the environment variable `TRANSFORMERS_VERBOSITY` to override the default verbosity. You can set it
to one of the following: `debug`, `info`, `warning`, `error`, `critical`. For example:

```bash
TRANSFORMERS_VERBOSITY=error ./myprogram.py
```

Additionally, some `warnings` can be disabled by setting the environment variable
`TRANSFORMERS_NO_ADVISORY_WARNINGS` to a true value, like *1*. This will disable any warning that is logged using
[`logger.warning_advice`]. For example:

```bash
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 ./myprogram.py
```

Here is an example of how to use the same logger as the library in your own module or script:

```python
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")
```


All the methods of this logging module are documented below, the main ones are
[`logging.get_verbosity`] to get the current level of verbosity in the logger and
[`logging.set_verbosity`] to set the verbosity to the level of your choice. In order (from the least
verbose to the most verbose), those levels (with their corresponding int values in parenthesis) are:

- `transformers.logging.CRITICAL` or `transformers.logging.FATAL` (int value, 50): only report the most
  critical errors.
- `transformers.logging.ERROR` (int value, 40): only report errors.
- `transformers.logging.WARNING` or `transformers.logging.WARN` (int value, 30): only reports error and
  warnings. This the default level used by the library.
- `transformers.logging.INFO` (int value, 20): reports error, warnings and basic information.
- `transformers.logging.DEBUG` (int value, 10): report all information.

By default, `tqdm` progress bars will be displayed during model download. [`logging.disable_progress_bar`] and [`logging.enable_progress_bar`] can be used to suppress or unsuppress this behavior.

## Base setters

[[autodoc]] logging.set_verbosity_error

[[autodoc]] logging.set_verbosity_warning

[[autodoc]] logging.set_verbosity_info

[[autodoc]] logging.set_verbosity_debug

## Other functions

[[autodoc]] logging.get_verbosity

[[autodoc]] logging.set_verbosity

[[autodoc]] logging.get_logger

[[autodoc]] logging.enable_default_handler

[[autodoc]] logging.disable_default_handler

[[autodoc]] logging.enable_explicit_format

[[autodoc]] logging.reset_format

[[autodoc]] logging.enable_progress_bar

[[autodoc]] logging.disable_progress_bar
