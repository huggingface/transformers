import ast
import json
import re
import jmespath  # TODO Make this a proper optional dependency
from collections import namedtuple

# Next line only used because eval might grab them. Can be removed once we have something better than eval
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: F401

from .chat_template_utils import _parse_type_hint, parse_google_format_docstring


offset_content = namedtuple("offset_content", ["content", "offset"])
location_content = namedtuple("location_content", ["content", "start", "end"])


def _parse_re_match(node_match, offset, require_groups: list[str] | None = None):
    if require_groups:
        if not node_match.groupdict():
            raise ValueError(f"Regex has no named groups, but require_groups was set to {require_groups}")
        for group in require_groups:
            if group not in node_match.groupdict():
                raise ValueError(f"Regex missing required group {group}!\nGroups: {node_match.groupdict().keys()}\n")
    # If the regex has named groups, return a dict of those groups
    if node_match.groupdict():
        if offset is not None:
            return {
                group: offset_content(content, node_match.start(group) + offset)
                for group, content in node_match.groupdict().items()
            }
        else:
            return node_match.groupdict()
    # If the regex has unnamed groups, it MUST only have one, and we return that group
    elif groups := list(node_match.groups()):
        if len(groups) > 1:
            raise ValueError(f"Regex has multiple unnamed groups!\nGroups: {groups}\n")
        if offset is not None:
            return offset_content(groups[0], node_match.start(1) + offset)  # TODO Is (1) right here?
        else:
            return groups[0]
    # If no groups, use the whole match
    else:
        if offset is not None:
            return offset_content(node_match.group(0), node_match.start(0) + offset)
        else:
            return node_match.group(0)


def recursive_parse(
    node_content: str | list | dict | offset_content,
    node_schema: dict,
    scope_vars: dict | None = None,
    offset: int | None = None,
):
    """
    This function takes content and a JSON schema node which includes
    regex extractors, and recursively parses the content according to the schema. It uses recursion to handle
    nested schemas.

    Args:
        node_content: The content corresponding to this node. Usually a string, but can be something else
                      if the parent node has multiple capture groups or named groups. In that case,
                      we generally pass the capture groups straight through to the children of this node
                      and don't do any parsing at this level.
        node_schema: The schema node controlling the parsing.
        scope_vars: A dictionary of variables in scope at the current node

    Returns:
        The parsed data structure for the current node.
    """

    # If the schema has a const, we just return that value and do absolutely nothing else
    if "const" in node_schema:
        return node_schema["const"]

    # If the node content is None, we return None. EZ.
    if node_content is None:
        return None

    # If we need to parse scope vars, now is the time to do it because they need to go to all child nodes
    if "x-scope-vars" in node_schema:
        if scope_vars is None:
            scope_vars = {}
        # Make sure we create a new dict so we don't accidentally send modifications up the tree
        scope_vars = scope_vars | {
            key: recursive_parse(node_content, value, scope_vars) for key, value in node_schema["x-scope-vars"].items()
        }

    # Next, if the node has a parser we can just use that and ignore everything else.
    if "x-parser" in node_schema:
        parser = node_schema["x-parser"]
        offset = None  # We cannot track offsets through parsers, so stop trying
        if parser == "json":
            parser_args = node_schema.get("x-parser-args", {})
            try:
                parsed_json = json.loads(node_content)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Node has JSON parser but could not parse its contents as JSON: {node_content}\nError: {e}"
                )
            if "transform" in parser_args:
                parsed_json = jmespath.search(parser_args["transform"], parsed_json)
            node_content = parsed_json
        elif parser == "python_type":
            # TODO eval is obviously enormously insecure and only used for prototyping here
            #      make a safer parser before merging
            node_content = _parse_type_hint(eval(node_content))
        elif parser == "python_function":
            parser_args = node_schema.get("x-parser-args", {})
            node_content = _extract_args_and_docstring_from_function_text(node_content, **parser_args)
        else:
            raise ValueError(f"Unknown parser {parser} for schema node: {node_schema}")

    # If not, we have to do a little parsing. First, set some vars and do basic validation
    node_type = node_schema["type"]
    has_regex = "x-regex" in node_schema or "x-regex-iterator" in node_schema or "x-regex-to-dict" in node_schema
    if has_regex and not isinstance(node_content, str):
        raise TypeError(
            "Schema node got a non-string input, but has a regex for parsing.\n"
            f"Input: {node_content}\n"
            f"Schema: {node_schema}"
        )

    node_regex = node_schema.get("x-regex")
    node_regex_iterator = node_schema.get("x-regex-iterator")
    node_regex_to_dict = node_schema.get("x-regex-to-dict")
    if node_regex is not None:
        node_match = re.search(node_regex, node_content, flags=re.DOTALL)
        if not node_match:
            return None  # TODO Is this correct? Should I raise an error?
        node_content = _parse_re_match(node_match, offset)
        if isinstance(node_content, offset_content):
            offset = node_content.offset
            node_content = node_content.content
    if node_regex_iterator is not None:
        if node_type != "array":
            raise TypeError(f"Schema node with type {node_type} cannot use x-regex-iterator.\nSchema: {node_schema}")
        # Note that this can be applied after a standard node-regex search
        node_content = [
            _parse_re_match(node_match, offset)
            for node_match in re.finditer(node_regex_iterator, node_content, flags=re.DOTALL)
        ]
        if not node_content:
            return None  # TODO Is this correct? Should I raise an error?
    if node_regex_to_dict is not None:
        if node_type != "object":
            raise TypeError(f"Schema node with type {node_type} cannot use x-regex-to-dict.\nSchema: {node_schema}")
        # Note that this can be applied after a standard node-regex search
        output_content = {}
        for node_match in re.finditer(node_regex_to_dict, node_content, flags=re.DOTALL):
            match_groups = _parse_re_match(node_match, offset, require_groups=["key", "value"])
            output_content[match_groups["key"]] = match_groups["value"]
        node_content = output_content
        if not node_content:
            return None

    # If there's a mapping, apply it now
    if "x-mapping" in node_schema:
        if not isinstance(node_content, str):
            raise TypeError(
                f"Schema node with type {node_type} cannot use x-mapping on non-string content.\n"
                f"Content: {node_content}\n"
                f"Schema: {node_schema}"
            )
        mapping = node_schema["x-mapping"]
        if node_content in mapping:
            node_content = mapping[node_content]
            offset = None  # Don't try to track offsets through mappings

    if "x-mapping-regex" in node_schema:
        if not isinstance(node_content, str):
            raise TypeError(
                f"Schema node with type {node_type} cannot use x-mapping-regex on non-string content.\n"
                f"Content: {node_content}\n"
                f"Schema: {node_schema}"
            )
        mapping_regex = node_schema["x-mapping-regex"]
        for pattern, replacement in mapping_regex.items():
            substitution = re.sub(pattern, replacement, node_content, flags=re.DOTALL)
            if substitution != node_content:
                offset = None  # If we make a substitution, we can't track offsets anymore
                node_content = substitution

    # Finally, handle parsed content based on schema type and recurse if required
    if node_type == "object":
        if not isinstance(node_content, (dict, str)):
            raise TypeError(f"Expected a dict or str for schema node with type object, got {node_content}")
        parsed_schema = {}
        if isinstance(node_content, str):
            # This means we don't have a regex at this level, so all of our child nodes need to parse the whole
            # string themselves to extract their value.
            if "properties" not in node_schema:
                raise ValueError(
                    f"Object node received string content but has no regex or parser to handle it.\n"
                    f"Content: {node_content}\n"
                    f"Schema: {node_schema}"
                )
            for key, child_node in node_schema["properties"].items():
                parsed_schema[key] = recursive_parse(node_content, node_schema["properties"][key], scope_vars, offset)
            return parsed_schema
        for key, child_node in node_schema.get("properties", {}).items():
            # TODO Error if required keys are not present
            if key in node_content:
                key_content = node_content[key]
                if isinstance(key_content, offset_content):
                    key_offset = key_content.offset
                    key_content = key_content.content
                else:
                    key_offset = offset
                parsed_schema[key] = recursive_parse(key_content, child_node, scope_vars, key_offset)
            elif "default" in child_node:
                # TODO Do I want to allow defaults?
                parsed_schema[key] = child_node["default"]
            else:
                pass  # TODO Add an error for required keys not present
        if "additionalProperties" in node_schema:
            for key, value in node_content.items():
                if key not in node_schema.get("properties", {}):
                    if isinstance(value, offset_content):
                        value_offset = value.offset
                        value = value.content
                    else:
                        value_offset = offset
                    parsed_schema[key] = recursive_parse(
                        value, node_schema["additionalProperties"], scope_vars, offset=value_offset
                    )
        return parsed_schema
    elif node_type == "array":
        if not node_content:
            return []
        parsed_schema = []
        if "items" in node_schema:
            if not isinstance(node_content, list):
                raise TypeError(f"Expected a list or regex for schema node with type array, got {node_content}")
            for item in node_content:
                if isinstance(item, offset_content):
                    item_content = item.content
                    item_offset = item.offset
                else:
                    item_content = item
                    item_offset = offset
                parsed_schema.append(recursive_parse(item_content, node_schema["items"], scope_vars, item_offset))
            return parsed_schema
        elif "prefixItems" in node_schema:
            if not isinstance(node_content, list):
                if len(node_schema["prefixItems"]) == 1:
                    # If there's only one prefix item, this is a single item array, we can just wrap the string
                    node_content = [node_content]
                else:
                    raise TypeError(f"Expected a list or regex for schema node with type array, got {node_content}")
            if len(node_content) != len(node_schema["prefixItems"]):
                raise ValueError(
                    f"Array node has {len(node_content)} items, but schema only has "
                    f"{len(node_schema['prefixItems'])} prefixItems defined.\n"
                    f"Content: {node_content}\n"
                    f"Schema: {node_schema}"
                )
            for item, item_schema in zip(node_content, node_schema["prefixItems"]):
                if isinstance(item, offset_content):
                    item_content = item.content
                    item_offset = item.offset
                else:
                    item_content = item
                    item_offset = offset
                parsed_schema.append(recursive_parse(item_content, item_schema, scope_vars, item_offset))
            return parsed_schema
        else:
            raise ValueError(f"Array node has no items or prefixItems schema defined.\nSchema: {node_schema}")
    elif node_type in ("string", "integer", "number", "boolean"):
        if not isinstance(node_content, str):
            raise TypeError(f"Expected a string for schema node with type {node_type}, got {node_content}")
        if node_type == "integer":
            if offset is not None:
                return location_content(int(node_content), offset, offset + len(node_content))
            else:
                return int(node_content)
        elif node_type == "number":
            if offset is not None:
                return location_content(float(node_content), offset, offset + len(node_content))
            else:
                return float(node_content)
        elif node_type == "boolean":
            if node_content.lower() in ("true", "1"):
                value = True
            elif node_content.lower() in ("false", "0"):
                value = False
            else:
                raise ValueError(f"Invalid boolean value: {node_content}")
            if offset is not None:
                return location_content(value, offset, offset + len(node_content))
            else:
                return value
        else:
            if offset is not None:
                return location_content(node_content, offset, offset + len(node_content))
            else:
                return node_content
    else:
        # TODO Should we handle null types?
        raise TypeError(f"Unsupported schema type {node_type} for node: {node_content}")


def _extract_args_and_docstring_from_function_text(function_text: str, include_return: bool = True) -> dict:
    tree = ast.parse(function_text)
    func = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
    if func is None:
        raise ValueError("Parser `python_function` couldn't find a function def: \n\n", function_text)

    docstring = ast.get_docstring(func)
    if docstring:
        description, args_dict, return_description = parse_google_format_docstring(docstring)
        args_dict = {key: {"description": value} for key, value in args_dict.items()}
    else:
        description = args_dict = return_description = None

    required_args = []
    # Right-align defaults to args
    defaults = [None] * (len(func.args.args) - len(func.args.defaults)) + func.args.defaults
    for arg, default in zip(func.args.args, defaults):
        if not default:
            required_args.append(arg.arg)
        # TODO This is a horrific temporary hack, write a better function to parse the AST and not eval the unparse
        #      like some kind of hobgoblin
        arg_type = _parse_type_hint(eval(ast.unparse(arg.annotation))) if arg.annotation else None
        if arg_type:
            args_dict[arg.arg] = args_dict.get(arg.arg, {})
            args_dict[arg.arg]["type"] = arg_type["type"]
    if func.returns:
        return_type = _parse_type_hint(eval(ast.unparse(func.returns)))
    else:
        return_type = None

    parameters_dict = {"type": "object", "properties": args_dict}
    if required_args:
        parameters_dict["required"] = required_args

    out = {"name": func.name, "description": description, "parameters": parameters_dict}
    if include_return and (return_type or return_description):
        returns = {}
        if return_type:
            returns["type"] = return_type["type"]
        if return_description:
            returns["description"] = return_description
        out["return"] = returns
    return out
