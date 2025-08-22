import re
import json
from .chat_template_utils import _parse_type_hint

# Next line only used because eval might grab them. Can be removed once we have something better than eval
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: F401

def recursive_parse(node_content: str | list | dict, node_schema: dict, scope_vars: dict = None):
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
    def _parse_re_match(node_match):
        # If the regex has named groups, return a dict of those groups
        if node_match.groupdict():
            return node_match.groupdict()
        # If the regex has unnamed groups, it MUST only have one, and we return that group
        elif groups := list(node_match.groups()):
            if len(groups) > 1:
                raise ValueError(f"Regex has multiple unnamed groups!\n"
                                 f"Groups: {groups}\n"
                                 )
            return groups[0]
        # If no groups, use the whole match
        else:
            return node_match.group(0)
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
        scope_vars = scope_vars | {key: recursive_parse(node_content, value, scope_vars)
                          for key, value in node_schema["x-scope-vars"].items()}

    # Next, if the node has a parser we can just use that and ignore everything else.
    if "x-parser" in node_schema:
        parser = node_schema["x-parser"]
        if parser == "json":
            try:
                return json.loads(node_content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Node has JSON parser but could not parse its contents as JSON: {node_content}\n"
                                 f"Error: {e}")
        elif parser == "python_type":
            # TODO eval is obviously enormously insecure and only used for prototyping here
            #      make a safer parser before merging
            return _parse_type_hint(eval(node_content))
        else:
            raise ValueError(f"Unknown parser {parser} for schema node: {node_schema}")



    # If not, we have to do a little parsing. First, set some vars and do basic validation
    node_type = node_schema["type"]
    has_regex = "x-regex" in node_schema or "x-regex-iterator" in node_schema or "x-regex-to-dict" in node_schema
    if not has_regex and isinstance(node_content, str) and node_type == "array":
        raise TypeError(f"array node got a string input, but has no parser or regex.\n"
                        f"Input: {node_content}\n",
                        f"Schema: {node_schema}")
    if has_regex and not isinstance(node_content, str):
        raise TypeError("Schema node got a non-string input, but has a regex for parsing.\n"
                        f"Input: {node_content}\n"
                        f"Schema: {node_schema}")


    node_regex = node_schema.get("x-regex")
    node_regex_iterator = node_schema.get("x-regex-iterator")
    node_regex_to_dict = node_schema.get("x-regex-to-dict")
    if node_regex is not None:
        node_match = re.search(node_regex, node_content, flags=re.DOTALL)
        if not node_match:
            return None  # TODO Is this correct? Should I raise an error?
        node_content = _parse_re_match(node_match)
    if node_regex_iterator is not None:
        if node_type != "array":
            raise TypeError(f"Schema node with type {node_type} cannot use x-regex-iterator.\n"
                            f"Schema: {node_schema}")
        # Note that this can be applied after a standard node-regex search
        node_content = [_parse_re_match(node_match) for node_match in re.finditer(node_regex_iterator, node_content, flags=re.DOTALL)]
        if not node_content:
            return None  # TODO Is this correct? Should I raise an error?
    if node_regex_to_dict is not None:
        if node_type != "object":
            raise TypeError(f"Schema node with type {node_type} cannot use x-regex-to-dict.\n"
                            f"Schema: {node_schema}")
        # Note that this can be applied after a standard node-regex search
        output_content = {}
        for node_match in re.finditer(node_regex_to_dict, node_content, flags=re.DOTALL):
            if not (match_keys := node_match.groupdict()):
                raise ValueError(f"Regex for x-regex-to-dict must return groups named \"key\" and \"value\".\n"
                                 f"Regex: {node_regex_to_dict}\n"
                                 f"Match: {node_match.group(0)}\n")
            if not set(match_keys.keys()) == {"key", "value"}:
                raise ValueError(f"Regex for x-regex-to-dict must return groups named \"key\" and \"value\".\n"
                                 f"Regex: {node_regex_to_dict}\n"
                                 f"Match: {node_match.group(0)}\n")
            output_content[match_keys["key"]] = match_keys["value"]
        node_content = output_content
        if not node_content:
            return None

    # Finally, handle parsed content based on schema type and recurse if required
    if node_type == "object":
        if not isinstance(node_content, (dict, str)):
            raise TypeError(f"Expected a dict or str for schema node with type object, got {node_content}")
        parsed_schema = {}
        if isinstance(node_content, str):
            # This means we don't have a regex at this level, so all of our child nodes need to parse the whole
            # string themselves to extract their value.
            for key, child_node in node_schema["properties"].items():
                parsed_schema[key] = recursive_parse(node_content, node_schema["properties"][key], scope_vars)
            return parsed_schema
        for key, child_node in node_schema.get("properties", {}).items():
            # TODO Error if required keys are not present
            if key in node_content:
                parsed_schema[key] = recursive_parse(node_content[key], child_node, scope_vars)
            elif "default" in child_node:
                # TODO Do I want to allow defaults?
                parsed_schema[key] = child_node["default"]
            else:
                pass  # TODO Add an error for required keys not present
        if "additionalProperties" in node_schema:
            # TODO Allow untyped additional properties with a parser where we just dump the entire parser output?
            for key, value in node_content.items():
                if key not in node_schema.get("properties", {}):
                    parsed_schema[key] = recursive_parse(value, node_schema["additionalProperties"], scope_vars)
        return parsed_schema
    elif node_type == "array":
        if not node_content:
            return []
        if not isinstance(node_content, list):
            raise TypeError(f"Expected a list or regex for schema node with type array, got {node_content}")
        parsed_schema = []
        # TODO Handle tuples/prefixItems?
        for item in node_content:
            parsed_schema.append(recursive_parse(item, node_schema["items"], scope_vars))
        return parsed_schema
    elif node_type in ("string", "integer", "number", "boolean"):
        if not isinstance(node_content, str):
            raise TypeError(f"Expected a string for schema node with type {node_type}, got {node_content}")
        if node_type == "integer":
            return int(node_content)
        elif node_type == "number":
            return float(node_content)
        elif node_type == "boolean":
            if node_content.lower() in ("true", "1"):
                return True
            elif node_content.lower() in ("false", "0"):
                return False
            else:
                raise ValueError(f"Invalid boolean value: {node_content}")
        return node_content
    else:
        # TODO Should we handle null types?
        raise TypeError(f"Unsupported schema type {node_type} for node: {node_type}")


def _split_at_top_level(s: str, delimiter: str) -> list[str]:
    """
    Splits a string by a delimiter, but only at the top level (not inside brackets).

    For example:
    _split_at_top_level("int | list[str, int]", "|") -> ["int", "list[str, int]"]
    _split_at_top_level("str, list[str, int]", ",") -> ["str", "list[str, int]"]

    Args:
        s: The string to split.
        delimiter: The character to split on.

    Returns:
        A list of substrings.
    """
    parts = []
    bracket_level = 0
    current_part_start_index = 0
    for i, char in enumerate(s):
        if char == '[':
            bracket_level += 1
        elif char == ']':
            bracket_level -= 1
            if bracket_level < 0:
                # This indicates a malformed string like "str]]"
                # We'll let it be handled as a simple name later.
                pass
        elif char == delimiter and bracket_level == 0:
            parts.append(s[current_part_start_index:i].strip())
            current_part_start_index = i + 1

    # Add the final part of the string
    parts.append(s[current_part_start_index:].strip())
    return parts


def parse_type_string(type_str: str) -> dict:
    """
    Recursively parses a Python type hint string into a structured dictionary.

    This parser specifically handles:
    - Simple types (e.g., 'int', 'str')
    - The union operator '|' (e.g., 'int | str')
    - Generic types like 'Optional[T]', 'Union[S, T]', and 'list[U]'

    It does NOT use `eval()` or `ast.parse()`.

    Args:
        type_str: The string representation of the type hint.

    Returns:
        A dictionary representing the parsed type structure.
    """
    # Clean up any leading/trailing whitespace
    type_str = type_str.strip()

    # 1. Handle unions with the '|' operator first, as it has lower precedence
    #    than the generic type syntax (e.g., list[...]).
    top_level_or_parts = _split_at_top_level(type_str, '|')
    if len(top_level_or_parts) > 1:
        return {
            "type": "union",
            "args": [parse_type_string(part) for part in top_level_or_parts]
        }

    # 2. Handle generic types (e.g., Optional[...], Union[...], list[...])
    if type_str.endswith(']'):
        try:
            first_bracket_index = type_str.index('[')
            base_type = type_str[:first_bracket_index].strip()

            # Extract the content within the brackets
            args_str = type_str[first_bracket_index + 1:-1].strip()

            if not args_str:
                # Handle cases like list[] with no arguments
                return {"type": "generic", "base": base_type, "args": []}

            # Split the arguments inside the brackets, respecting nested structures
            arg_parts = _split_at_top_level(args_str, ',')

            # Special handling for Union[...] syntax
            if base_type == "Union":
                return {
                    "type": "union",
                    "args": [parse_type_string(part) for part in arg_parts]
                }

            # Special handling for Optional[...] syntax
            if base_type == "Optional":
                if len(arg_parts) != 1:
                    raise ValueError("Parser error: Optional[] must have exactly one argument.")
                # We can represent it as a specific "optional" type or as a union with None.
                # Here, we'll keep it distinct.
                return {
                    "type": "generic",
                    "base": "Optional",
                    "args": [parse_type_string(arg_parts[0])]
                }

            # Handle all other generic types like list, dict, etc.
            return {
                "type": "generic",
                "base": base_type,
                "args": [parse_type_string(part) for part in arg_parts]
            }
        except ValueError:
            # This can happen if '[' is not found or if the string is malformed.
            # In such cases, we fall through to treat it as a simple name.
            pass

    # 3. Base case: The string is a simple type name (e.g., 'int', 'str', 'MyClass')
    return {
        "type": "name",
        "value": type_str
    }