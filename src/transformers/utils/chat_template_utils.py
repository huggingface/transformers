import inspect
import re
from typing import Any, Union, get_origin, get_type_hints
import pdb


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
    required = [
        param_name for param_name, param in signature.parameters.items() if param.default == inspect.Parameter.empty
    ]

    for param_name, param_type in type_hints.items():
        if param_name == "return":
            continue
        pdb.set_trace()
        if (origin := get_origin(param_type)) is not None:
            if origin is Union:
                if all(t in BASIC_TYPES for t in param_type.__args__):
                    properties[param_name] = {
                        "type": [_get_json_schema_type(t)["type"] for t in param_type.__args__ if t != type(None)],
                        "nullable": type(None) in param_type.__args__,
                    }
                else:
                    properties[param_name] = {
                        "anyOf": [_get_json_schema_type(t) for t in param_type.__args__ if t != type(None)],
                        "nullable": type(None) in param_type.__args__,
                    }
            elif origin is list:
                properties[param_name] = {"type": "array", "items": _get_json_schema_type(param_type.__args__[0])}
            elif origin is dict:
                properties[param_name] = {
                    "type": "object",
                    "additionalProperties": _get_json_schema_type(param_type.__args__[1]),
                }
        else:
            properties[param_name] = _get_json_schema_type(param_type)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema


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
