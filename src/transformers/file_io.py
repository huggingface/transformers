"""Utilities for simplifying the common Files IO."""

from typing import Any, Text

import jsonpickle


DEFAULT_ENCODING = "utf-8"


def json_pickle_dump(filename: Text, obj: Any) -> None:
    """
    Pickle an object into JSON decodable representation and save it to a file.

    Args:
        filename (:obj:`Any`): The file used to save the pickled object.
        obj (:obj:`Any`): The object to pickle into JSON decodable representation.
    """
    with open(filename, mode="w", encoding=DEFAULT_ENCODING) as output_file:
        output_file.write(jsonpickle.encode(obj))


def json_pickle_load(filename: Text) -> Any:
    """
    Load the pickled JSON object from a file and unpickle it.

    Args:
        json_obj (:obj:`Text`): The file used to load the pickled object.

    Returns:
        :obj:`Any`: An unpickled object.
    """
    with open(filename, encoding=DEFAULT_ENCODING) as input_file:
        return jsonpickle.decode(input_file.read())
