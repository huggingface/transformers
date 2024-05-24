import functools
import inspect
import warnings
from typing import Optional


def filter_out_non_signature_kwargs(extra: Optional[list] = None):
    """Filter out named arguments that are not in the function signature."""

    extra = extra or []
    extra_params_to_pass = set(extra)

    def decorator(func):
        sig = inspect.signature(func)
        function_named_args = set(sig.parameters.keys())
        valid_kwargs_to_pass = function_named_args.union(extra_params_to_pass)
        is_class_method = "self" in function_named_args

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            valid_kwargs = {}
            invalid_kwargs = {}

            for k, v in kwargs.items():
                if k in valid_kwargs_to_pass:
                    valid_kwargs[k] = v
                else:
                    invalid_kwargs[k] = v

            if invalid_kwargs:
                invalid_kwargs_names = ", ".join(invalid_kwargs.keys())
                cls_name = args[0].__class__.__name__ + "." if is_class_method else ""
                warnings.warn(
                    f"The following named arguments are not valid for `{cls_name}{func.__name__}`"
                    f" and were ignored: {invalid_kwargs_names}"
                )

            return func(*args, **valid_kwargs)

        return wrapper

    return decorator
