# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import inspect
import warnings
from functools import wraps
from typing import Optional

import packaging.version

from .. import __version__
from . import ExplicitEnum, is_torch_available, is_torchdynamo_compiling


# This is needed in case we deprecate a kwarg of a function/method being compiled
if is_torch_available():
    import torch  # noqa: F401


class Action(ExplicitEnum):
    NONE = "none"
    NOTIFY = "notify"
    NOTIFY_ALWAYS = "notify_always"
    RAISE = "raise"


def deprecate_kwarg(
    old_name: str,
    version: str,
    new_name: Optional[str] = None,
    warn_if_greater_or_equal_version: bool = False,
    raise_if_greater_or_equal_version: bool = False,
    raise_if_both_names: bool = False,
    additional_message: Optional[str] = None,
):
    """
    Function or method decorator to notify users about deprecated keyword arguments, replacing them with a new name if specified.
    Note that is decorator is `torch.compile`-safe, i.e. it will not cause graph breaks (but no warning will be displayed if compiling).

    This decorator allows you to:
    - Notify users when a keyword argument is deprecated.
    - Automatically replace deprecated keyword arguments with new ones.
    - Raise an error if deprecated arguments are used, depending on the specified conditions.

    By default, the decorator notifies the user about the deprecated argument while the `transformers.__version__` < specified `version`
    in the decorator. To keep notifications with any version `warn_if_greater_or_equal_version=True` can be set.

    Parameters:
        old_name (`str`):
            Name of the deprecated keyword argument.
        version (`str`):
            The version in which the keyword argument was (or will be) deprecated.
        new_name (`Optional[str]`, *optional*):
            The new name for the deprecated keyword argument. If specified, the deprecated keyword argument will be replaced with this new name.
        warn_if_greater_or_equal_version (`bool`, *optional*, defaults to `False`):
            Whether to show warning if current `transformers` version is greater or equal to the deprecated version.
        raise_if_greater_or_equal_version (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if current `transformers` version is greater or equal to the deprecated version.
        raise_if_both_names (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if both deprecated and new keyword arguments are set.
        additional_message (`Optional[str]`, *optional*):
            An additional message to append to the default deprecation message.

    Raises:
        ValueError:
            If raise_if_greater_or_equal_version is True and the current version is greater than or equal to the deprecated version, or if raise_if_both_names is True and both old and new keyword arguments are provided.

    Returns:
        Callable:
            A wrapped function that handles the deprecated keyword arguments according to the specified parameters.

    Example usage with renaming argument:

        ```python
        @deprecate_kwarg("reduce_labels", new_name="do_reduce_labels", version="6.0.0")
        def my_function(do_reduce_labels):
            print(do_reduce_labels)

        my_function(reduce_labels=True)  # Will show a deprecation warning and use do_reduce_labels=True
        ```

    Example usage without renaming argument:

        ```python
        @deprecate_kwarg("max_size", version="6.0.0")
        def my_function(max_size):
            print(max_size)

        my_function(max_size=1333)  # Will show a deprecation warning
        ```

    """

    deprecated_version = packaging.version.parse(version)
    current_version = packaging.version.parse(__version__)
    is_greater_or_equal_version = current_version >= deprecated_version

    if is_greater_or_equal_version:
        version_message = f"and removed starting from version {version}"
    else:
        version_message = f"and will be removed in version {version}"

    def wrapper(func):
        # Required for better warning message
        sig = inspect.signature(func)
        function_named_args = set(sig.parameters.keys())
        is_instance_method = "self" in function_named_args
        is_class_method = "cls" in function_named_args

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # Get class + function name (just for better warning message)
            func_name = func.__name__
            if is_instance_method:
                func_name = f"{args[0].__class__.__name__}.{func_name}"
            elif is_class_method:
                func_name = f"{args[0].__name__}.{func_name}"

            minimum_action = Action.NONE
            message = None

            # deprecated kwarg and its new version are set for function call -> replace it with new name
            if old_name in kwargs and new_name in kwargs:
                minimum_action = Action.RAISE if raise_if_both_names else Action.NOTIFY_ALWAYS
                message = f"Both `{old_name}` and `{new_name}` are set for `{func_name}`. Using `{new_name}={kwargs[new_name]}` and ignoring deprecated `{old_name}={kwargs[old_name]}`."
                kwargs.pop(old_name)

            # only deprecated kwarg is set for function call -> replace it with new name
            elif old_name in kwargs and new_name is not None and new_name not in kwargs:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message} for `{func_name}`. Use `{new_name}` instead."
                kwargs[new_name] = kwargs.pop(old_name)

            # deprecated kwarg is not set for function call and new name is not specified -> just notify
            elif old_name in kwargs:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message} for `{func_name}`."

            if message is not None and additional_message is not None:
                message = f"{message} {additional_message}"

            # update minimum_action if argument is ALREADY deprecated (current version >= deprecated version)
            if is_greater_or_equal_version:
                # change to (NOTIFY, NOTIFY_ALWAYS) -> RAISE if specified
                # in case we want to raise error for already deprecated arguments
                if raise_if_greater_or_equal_version and minimum_action != Action.NONE:
                    minimum_action = Action.RAISE

                # change to NOTIFY -> NONE if specified (NOTIFY_ALWAYS can't be changed to NONE)
                # in case we want to ignore notifications for already deprecated arguments
                elif not warn_if_greater_or_equal_version and minimum_action == Action.NOTIFY:
                    minimum_action = Action.NONE

            # raise error or notify user
            if minimum_action == Action.RAISE:
                raise ValueError(message)
            # If we are compiling, we do not raise the warning as it would break compilation
            elif minimum_action in (Action.NOTIFY, Action.NOTIFY_ALWAYS) and not is_torchdynamo_compiling():
                # DeprecationWarning is ignored by default, so we use FutureWarning instead
                warnings.warn(message, FutureWarning, stacklevel=2)

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper
