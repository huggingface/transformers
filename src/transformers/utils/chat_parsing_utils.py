import re

# TODO Cases to handle:
#      1) Model has multiple chat templates

def recursive_parse(node_content: str | list | dict, node_schema: dict):
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
    # First, set some vars and do basic validation
    node_type = node_schema["type"]
    has_regex = "x-regex" in node_schema or "x-regex-iterator" in node_schema
    if not has_regex and isinstance(node_content, str) and node_type == "array":
        raise TypeError(f"array node got a string input, but has no regex for parsing.\n"
                        f"Input: {node_content}")
    if has_regex and not isinstance(node_content, str):
        raise TypeError("Schema node got a non-string input, but has a regex for parsing.\n"
                        f"Input: {node_content}\n"
                        f"Schema: {node_schema}")


    if node_schema.get("x-regex", None) is not None:
        node_regex = node_schema["x-regex"]
        node_match = re.search(node_regex, node_content, flags=re.DOTALL)
        if not node_match:
            return None  # TODO Is this correct? Should I raise an error?
        node_content = _parse_re_match(node_match)
    elif node_schema.get("x-regex-iterator", None) is not None:
        node_regex_iterator = node_schema["x-regex-iterator"]
        node_content = [_parse_re_match(node_match) for node_match in re.finditer(node_regex_iterator, node_content, flags=re.DOTALL)]

        if not node_content:
            return None  # TODO Is this correct? Should I raise an error?

    # Finally, handle parsed content based on schema type and recurse if required
    if node_type == "object":
        if not isinstance(node_content, (dict, str)):
            raise TypeError(f"Expected a dict or str for schema node with type object, got {node_content}")
        parsed_schema = {}
        if isinstance(node_content, str):
            # This means we don't have a regex at this level, so all of our child nodes need to parse the whole
            # string themselves to extract their value.
            for key, child_node in node_schema["properties"].items():
                parsed_schema[key] = recursive_parse(node_content, node_schema["properties"][key])
        for key, child_node in node_schema["properties"].items():
            # TODO Error if required keys are not present
            if key in node_content:
                parsed_schema[key] = recursive_parse(node_content[key], child_node)
            elif "default" in child_node:
                # TODO Do I want to allow defaults?
                parsed_schema[key] = child_node["default"]
            else:
                pass  # TODO Add an error for required keys not present
        return parsed_schema
    elif node_type == "array":
        if not isinstance(node_content, list):
            raise TypeError(f"Expected a list or regex for schema node with type array, got {node_content}")
        parsed_schema = []
        # TODO Handle tuples/prefixItems?
        for item in node_content:
            parsed_schema.append(recursive_parse(item, node_schema["items"]))
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
