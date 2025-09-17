import json
import re

from transformers.utils import is_jmespath_available


if is_jmespath_available():
    import jmespath
else:
    jmespath = None


def _parse_re_match(node_match, require_groups: list[str] | None = None):
    if require_groups:
        if not node_match.groupdict():
            raise ValueError(f"Regex has no named groups, but require_groups was set to {require_groups}")
        for group in require_groups:
            if group not in node_match.groupdict():
                raise ValueError(f"Regex missing required group {group}!\nGroups: {node_match.groupdict().keys()}\n")
    # If the regex has named groups, return a dict of those groups
    if node_match.groupdict():
        return {key: val for key, val in node_match.groupdict().items() if val is not None}
    # If the regex has unnamed groups, it MUST only have one, and we return that group
    elif groups := list(node_match.groups()):
        if len(groups) > 1:
            raise ValueError(f"Regex has multiple unnamed groups!\nGroups: {groups}\n")
        return groups[0]
    # If no groups, use the whole match
    else:
        return node_match.group(0)


def recursive_parse(
    node_content: str | list | dict,
    node_schema: dict,
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

    Returns:
        The parsed data structure for the current node.
    """

    # If the schema has a const, we just return that value and do absolutely nothing else
    if "const" in node_schema:
        return node_schema["const"]

    # If the node content is None, we return None. EZ.
    if node_content is None:
        return None

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
            return None
        node_content = _parse_re_match(node_match)
    if node_regex_iterator is not None:
        if node_type != "array":
            raise TypeError(f"Schema node with type {node_type} cannot use x-regex-iterator.\nSchema: {node_schema}")
        # Note that this can be applied after a standard node-regex search
        node_content = [
            _parse_re_match(node_match)
            for node_match in re.finditer(node_regex_iterator, node_content, flags=re.DOTALL)
        ]
        if not node_content:
            return None
    if node_regex_to_dict is not None:
        if node_type != "object":
            raise TypeError(f"Schema node with type {node_type} cannot use x-regex-to-dict.\nSchema: {node_schema}")
        # Note that this can be applied after a standard node-regex search
        output_content = {}
        for node_match in re.finditer(node_regex_to_dict, node_content, flags=re.DOTALL):
            match_groups = _parse_re_match(node_match, require_groups=["key", "value"])
            output_content[match_groups["key"]] = match_groups["value"]
        node_content = output_content
        if not node_content:
            return None

    # Next, if the node has a parser, apply it. We do this after regexes so that the regex can extract
    # a substring to parse, if needed.
    if "x-parser" in node_schema:
        parser = node_schema["x-parser"]
        if parser == "json":
            if not isinstance(node_content, str):
                raise TypeError(
                    f"Node has JSON parser but got non-string input: {node_content}\nSchema: {node_schema}"
                )
            parser_args = node_schema.get("x-parser-args", {})
            try:
                parsed_json = json.loads(node_content)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Node has JSON parser but could not parse its contents as JSON: {node_content}\nError: {e}"
                )
            if "transform" in parser_args:
                if jmespath is None:
                    raise ImportError(
                        "Chat response schema includes a jmespath transformation, but jmespath is not installed. You can install it with `pip install jmespath`."
                    )
                parsed_json = jmespath.search(parser_args["transform"], parsed_json)
            node_content = parsed_json
        else:
            raise ValueError(f"Unknown parser {parser} for schema node: {node_schema}")

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

    if "x-mapping-regex" in node_schema:
        if not isinstance(node_content, str):
            raise TypeError(
                f"Schema node with type {node_type} cannot use x-mapping-regex on non-string content.\n"
                f"Content: {node_content}\n"
                f"Schema: {node_schema}"
            )
        mapping_regex = node_schema["x-mapping-regex"]
        for pattern, replacement in mapping_regex.items():
            node_content = re.sub(pattern, replacement, node_content, flags=re.DOTALL)

    # Finally, handle parsed content based on schema type and recurse if required
    if node_type == "object":
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
                child_node_content = recursive_parse(node_content, node_schema["properties"][key])
                if child_node_content is not None:
                    parsed_schema[key] = child_node_content
            return parsed_schema
        elif isinstance(node_content, dict):
            for key, child_node in node_schema.get("properties", {}).items():
                # TODO Error if required keys are not present
                if key in node_content:
                    parsed_schema[key] = recursive_parse(node_content[key], child_node)
                elif "default" in child_node:
                    # TODO Do I want to allow defaults?
                    parsed_schema[key] = child_node["default"]
                else:
                    pass  # TODO Add an error for required keys not present
            if "additionalProperties" in node_schema:
                for key, value in node_content.items():
                    if key not in node_schema.get("properties", {}):
                        parsed_schema[key] = recursive_parse(value, node_schema["additionalProperties"])
            return parsed_schema
        else:
            raise TypeError(f"Expected a dict or str for schema node with type object, got {node_content}")
    elif node_type == "array":
        if not node_content:
            return []
        parsed_schema = []
        if "items" in node_schema:
            if not isinstance(node_content, list):
                raise TypeError(f"Expected a list or regex for schema node with type array, got {node_content}")
            for item in node_content:
                parsed_schema.append(recursive_parse(item, node_schema["items"]))
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
                parsed_schema.append(recursive_parse(item, item_schema))
            return parsed_schema
        else:
            raise ValueError(f"Array node has no items or prefixItems schema defined.\nSchema: {node_schema}")
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
        else:
            # String type
            return node_content
    elif node_type == "any":
        # TODO Is there a better way of handling this? Not sure if 'any' is in the spec, but we need something like it
        return node_content
    else:
        raise TypeError(f"Unsupported schema type {node_type} for node: {node_content}")
