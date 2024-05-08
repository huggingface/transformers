import inspect
import re
from typing import Any, Union, get_args, get_origin, get_type_hints


BASIC_TYPES = (int, float, str, bool, Any)


def get_json_schema(func):
    doc = inspect.getdoc(func)
    if not doc:
        raise ValueError(f"Cannot generate JSON schema for {func.__name__} because it has no docstring!")
    doc = doc.strip()
    main_doc, param_descriptions = _get_argument_descriptions_from_docstring(doc)

    json_schema = _convert_type_hints_to_json_schema(func)
    for arg in json_schema["properties"]:
        if arg not in param_descriptions:
            raise ValueError(
                f"Cannot generate JSON schema for {func.__name__} because the docstring has no description for the argument '{arg}'"
            )
        json_schema["properties"][arg]["description"] = param_descriptions[arg]

    return {"name": func.__name__, "description": main_doc, "parameters": json_schema}


def _get_argument_descriptions_from_docstring(doc):
    param_pattern = r":param (\w+): (.+)"
    params = re.findall(param_pattern, doc)
    main_doc = doc.split(":param")[0].strip()
    return main_doc, dict(params)


def _convert_type_hints_to_json_schema(func):
    type_hints = get_type_hints(func)
    properties = {}

    signature = inspect.signature(func)
    required = []
    for param_name, param in signature.parameters.items():
        if param.annotation == inspect.Parameter.empty:
            raise ValueError(f"Argument {param.name} is missing a type hint in function {func.__name__}")
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    for param_name, param_type in type_hints.items():
        if param_name == "return":
            continue
        properties[param_name] = _parse_type_hint(param_type)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema


def _parse_type_hint(hint):
    if (origin := get_origin(hint)) is not None:
        if origin is Union:
            # If it's a union of basic types, we can express that as a simple list in the schema
            if all(t in BASIC_TYPES for t in get_args(hint)):
                return_dict = {
                    "type": [_get_json_schema_type(t)["type"] for t in get_args(hint) if t not in (type(None), ...)]
                }
                if len(return_dict["type"]) == 1:
                    return_dict["type"] = return_dict["type"][0]
            else:
                # A union of more complex types requires us to recurse into each subtype
                return_dict = {
                    "anyOf": [_parse_type_hint(t) for t in get_args(hint) if t not in (type(None), ...)],
                }
                if len(return_dict["anyOf"]) == 1:
                    return_dict = return_dict["anyOf"][0]
            if type(None) in get_args(hint):
                return_dict["nullable"] = True
            return return_dict
        elif origin is list:
            if not get_args(hint):
                return {"type": "array"}
            if all(t in BASIC_TYPES for t in get_args(hint)):
                # Similarly to unions, a list of basic types can be expressed as a list in the schema
                items = {"type": [_get_json_schema_type(t)["type"] for t in get_args(hint) if t != type(None)]}
                if len(items["type"]) == 1:
                    items["type"] = items["type"][0]
            else:
                # And a list of more complex types requires us to recurse into each subtype again
                items = {"anyOf": [_parse_type_hint(t) for t in get_args(hint) if t not in (type(None), ...)]}
                if len(items["anyOf"]) == 1:
                    items = items["anyOf"][0]
            return_dict = {"type": "array", "items": items}
            if type(None) in get_args(hint):
                return_dict["nullable"] = True
            return return_dict
        elif origin is tuple:
            raise ValueError(
                "This helper does not parse Tuple types, as they are usually used to indicate that "
                "each position is associated with a specific type, and this requires JSON schemas "
                "that are not supported by most templates. We recommend "
                "either using List or List[Union] instead for arguments where this is appropriate, or "
                "splitting arguments with Tuple types into multiple arguments that take single inputs."
            )
        elif origin is dict:
            # The JSON equivalent to a dict is 'object', which mandates that all keys are strings
            # However, we can specify the type of the dict values with "additionalProperties"
            return {
                "type": "object",
                "additionalProperties": _parse_type_hint(get_args(hint)[1]),
            }
        else:
            raise ValueError("Couldn't parse this type hint, likely due to a custom class or object: ", hint)
    else:
        return _get_json_schema_type(hint)


def _get_json_schema_type(param_type):
    if param_type == int:
        return {"type": "integer"}
    elif param_type == float:
        return {"type": "number"}
    elif param_type == str:
        return {"type": "string"}
    elif param_type == bool:
        return {"type": "boolean"}
    elif param_type == Any:
        return {}
    else:
        return {"type": "object"}
