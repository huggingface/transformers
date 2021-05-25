from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Union


OnnxVariable = NamedTuple(
    "OnnxVariable",
    [("name", str), ("axes", Dict[int, str]), ("repeated", Union[int, str]), ("value", Optional[List[int]])],
)


@dataclass
class OnnxConfig:
    """
    Base class for ONNX exportable model describing metadata on how to export the model through the ONNX format.
    """

    # Input mapping of the form "input_name": {axis_id: "axis_name"}
    # example: {"input_ids": {0: "batch", 1: "sequence"}}
    # We use a list because the ordering of the items is VERY important
    inputs: List[OnnxVariable]

    # Output mapping of the form "output_name": {axis_id: "axis_name"}
    # example: {"last_hidden_layer": {0: "batch", 1: "sequence"}}
    # We use a list because the ordering of the items is VERY important
    outputs: List[OnnxVariable]

    # Define all the configuration keys we need to override before forwarding through the model
    runtime_config_overrides: Optional[Dict[str, Any]]

    # Does the model requires using external data format (i.e. model size > 2Gb)
    use_external_data_format: bool

    # Minimum required ONNX opset
    minimum_required_onnx_opset: int

    # ONNXRuntime provides model specific optimizer for some topologies
    # This one indicate which provider (if any) to use
    optimizer: Optional[str]

    # If optimizer is present, this set indicates which features to enable/disable when optimizing
    optimizer_features: Optional[Dict[str, bool]]

    # Optimizer parameters which can only be known at runtime
    optimizer_additional_args: Optional[Dict[str, Union[int, str]]]
