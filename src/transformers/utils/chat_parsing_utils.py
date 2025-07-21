import re

# TODO Need to clarify what's getting passed around - is "text" a string or a dict of capture groups?
#      At the root it's a string, but maybe we can make that a dict with a single entry?
#      A regex can either have one capture group, multiple numbered groups, or multiple named groups
# TODO Cases to handle:
#      1) Model has multiple chat templates
#      2) Alternate RE flags
#      3) Supporting possessive operators

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
    # First, do some basic validation
    has_regex = hasattr(node_schema, "x-regex") or hasattr(node_schema, "x-regex-iterator")
    if has_regex and isinstance(node_content, str):
        raise TypeError("Schema node got a string input, but has no regex for parsing.")
    elif not has_regex and not isinstance(node_content, str):
        raise TypeError("Schema node got a non-string input, but has a regex for parsing.")

    # Next, if we have a regex, apply it to parse node_content
    if node_regex := getattr(node_schema, "x-regex", None) is not None:
        node_match = re.search(node_regex, node_content)
        if not node_match:
            return None  # TODO Is this correct? Should I raise an error?
        # If the regex has named groups, return a dict of those groups
        if node_match.groupdict():
            node_content = node_match.groupdict()
        # If the regex has numbered groups, return a list of those groups
        # TODO Is there ambiguity between a string match and a list match with a single element?
        elif node_match.groups():
            node_content = list(node_match.groups())
        # If no groups, use the whole match
        else:
            node_content = node_match.group(0)

    elif node_regex := getattr(node_schema, "x-regex-iterator", None) is not None:
        # TODO Parse groups in here too
        node_content = list(re.finditer(node_regex, node_content))

    # Finally, handle parsed content based on schema type and recurse if required
    if node_type := node_schema["type"] == "object":
        if not isinstance(node_content, dict):
            raise TypeError(f"Expected a dict for schema node {node_schema['title']}, got {type(node_content)}")
        parsed_schema = {}
        for key, child_node in node_schema["properties"].items():
            # TODO Error if required keys are not present
            if key in node_content:
                parsed_schema[key] = recursive_parse(node_content[key], child_node)
            elif "default" in child_node:
                # TODO Do I want to allow defaults?
                parsed_schema[key] = child_node["default"]
        return parsed_schema
    elif node_type == "array":
        if not isinstance(node_content, list):
            raise TypeError(f"Expected a list for schema node {node_schema['title']}, got {type(node_content)}")
        parsed_schema = []
        # TODO Handle tuples/prefixItems?
        for item in node_content:
            parsed_schema.append(recursive_parse(item, node_schema["items"]))
        return parsed_schema
    elif node_type in ("string", "integer", "number", "boolean"):
        if not isinstance(node_content, str):
            raise TypeError(f"Expected a string for schema node {node_schema['title']}, got {type(node_content)}")
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
        raise TypeError(f"Unsupported schema type {node_type} for node {node_schema['title']}")
