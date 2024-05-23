import packaging.version

from functools import wraps
from typing import Optional
from warnings import warn


from . import ExplicitEnum
from .. import __version__


class Action(ExplicitEnum):
    RAISE = "raise"
    NOTIFY = "notify"


def warning(message):
    warn(message, FutureWarning)


def deprecate_kwarg(
    old_name: str,
    version: str,
    new_name: Optional[str] = None,
    raise_if_ge_version: bool = False,
    raise_if_both_names: bool = False,
    add_message: Optional[str] = None,
):
    """Function or method decorator to notify user about deprecated keyword argument and replace it with new name if specified.
    If new name is specified, deprecated keyword argument will be replaced with it.
    If new name is not specified, only notification will be shown.
    If raise_error is set to True, ValueError will be raised when deprecated keyword argument is used.

    Args:
        old_name (str): name of the deprecated keyword argument
        version (str): version when the keyword argument was (will be) deprecated
        new_name (Optional[str], optional): new name of the keyword argument. Defaults to None.
        raise_if_ge_version (bool, optional): raise ValueError if deprecated version is greater or equal to current version. Defaults to False.
        raise_if_both_names (bool, optional): raise ValueError if both deprecated and new keyword arguments are set. Defaults to False.

    Raises:
        ValueError: raised when deprecated keyword argument is used and `raise_error` is set to True
    """
    
    deprecated_version = packaging.version.parse(version)
    current_version = packaging.version.parse(__version__)
    is_already_deprecated = current_version >= deprecated_version

    if is_already_deprecated:
        version_message = f"and removed starting from version {version}"
    else:
        version_message = f"and will be removed in version {version}"

    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):

            minimum_action = None
            message = None
            
            # deprecated kwarg and its new version are set for function call -> replace it with new name
            if old_name in kwargs and new_name is not None and new_name in kwargs:
                minimum_action = Action.RAISE if raise_if_both_names else Action.NOTIFY
                message = f"Both `{old_name}` and `{new_name}` are set. Using `{new_name}={kwargs[new_name]}` and ignoring deprecated `{old_name}={kwargs[old_name]}`."
                kwargs.pop(old_name)

            # only deprecated kwarg is set for function call -> replace it with new name
            elif old_name in kwargs and new_name is not None and new_name not in kwargs:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message}. Use `{new_name}` instead."
                kwargs[new_name] = kwargs.pop(old_name)

            # deprecated kwarg is not set for function call and new name is not specified -> just notify
            elif old_name in kwargs:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message}."

            if message is not None and add_message is not None:
                message = f"{message} {add_message}"

            # raise error or notify user
            if minimum_action == Action.RAISE:
                raise ValueError(message)
            elif minimum_action == Action.NOTIFY and raise_if_ge_version and is_already_deprecated:
                raise ValueError(message)
            elif minimum_action == Action.NOTIFY:
                warning(message)

            return func(*args, **kwargs)

        return wrapped_func
            
    return wrapper
