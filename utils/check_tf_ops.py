# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

from tensorflow.core.protobuf.saved_model_pb2 import SavedModel


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_copies.py
REPO_PATH = "."

# Internal TensorFlow ops that can be safely ignored (mostly specific to a saved model)
INTERNAL_OPS = [
    "Assert",
    "AssignVariableOp",
    "EmptyTensorList",
    "MergeV2Checkpoints",
    "ReadVariableOp",
    "ResourceGather",
    "RestoreV2",
    "SaveV2",
    "ShardedFilename",
    "StatefulPartitionedCall",
    "StaticRegexFullMatch",
    "VarHandleOp",
]


def onnx_compliancy(saved_model_path, strict, opset):
    saved_model = SavedModel()
    onnx_ops = []

    with open(os.path.join(REPO_PATH, "utils", "tf_ops", "onnx.json")) as f:
        onnx_opsets = json.load(f)["opsets"]

    for i in range(1, opset + 1):
        onnx_ops.extend(onnx_opsets[str(i)])

    with open(saved_model_path, "rb") as f:
        saved_model.ParseFromString(f.read())

    model_op_names = set()

    # Iterate over every metagraph in case there is more than one (a saved model can contain multiple graphs)
    for meta_graph in saved_model.meta_graphs:
        # Add operations in the graph definition
        model_op_names.update(node.op for node in meta_graph.graph_def.node)

        # Go through the functions in the graph definition
        for func in meta_graph.graph_def.library.function:
            # Add operations in each function
            model_op_names.update(node.op for node in func.node_def)

    # Convert to list, sorted if you want
    model_op_names = sorted(model_op_names)
    incompatible_ops = []

    for op in model_op_names:
        if op not in onnx_ops and op not in INTERNAL_OPS:
            incompatible_ops.append(op)

    if strict and len(incompatible_ops) > 0:
        raise Exception(f"Found the following incompatible ops for the opset {opset}:\n" + incompatible_ops)
    elif len(incompatible_ops) > 0:
        print(f"Found the following incompatible ops for the opset {opset}:")
        print(*incompatible_ops, sep="\n")
    else:
        print(f"The saved model {saved_model_path} can properly be converted with ONNX.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_path", help="Path of the saved model to check (the .pb file).")
    parser.add_argument(
        "--opset", default=12, type=int, help="The ONNX opset against which the model has to be tested."
    )
    parser.add_argument(
        "--framework", choices=["onnx"], default="onnx", help="Frameworks against which to test the saved model."
    )
    parser.add_argument(
        "--strict", action="store_true", help="Whether make the checking strict (raise errors) or not (raise warnings)"
    )
    args = parser.parse_args()

    if args.framework == "onnx":
        onnx_compliancy(args.saved_model_path, args.strict, args.opset)
