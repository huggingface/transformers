from pathlib import Path
from typing import Any, Dict, Iterable


def generate_identified_filename(filename: Path, identifier: str) -> Path:
    """
    Append a string-identifier at the end (before the extension, if any) to the provided filepath

    Args:
        filename: pathlib.Path The actual path object we would like to add an identifier suffix
        identifier: The suffix to add

    Returns:
        (str) With concatenated identifier at the end of the filename
    """
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)


def flatten_output_collection_property(name: str, field: Iterable[Any]) -> Dict[str, Any]:
    """
    Flatten any potential nested structure expanding the name of the field with the index of the element within the
    structure.

    Args:
        name: The name of the nested structure
        field: The structure to, potentially, be flattened

    Returns:
        (Dict[str, Any]): Outputs with flattened structure and key mapping this new structure.

    """
    from itertools import chain

    return {f"{name}.{idx}": item for idx, item in enumerate(chain.from_iterable(field))}
