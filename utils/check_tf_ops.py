from tensorflow.core.protobuf.saved_model_pb2 import SavedModel


saved_model = SavedModel()
with open("here/saved_model/1/saved_model.pb", "rb") as f:
    saved_model.ParseFromString(f.read())
model_op_names = set()
# Iterate over every metagraph in case there is more than one
for meta_graph in saved_model.meta_graphs:
    # Add operations in the graph definition
    model_op_names.update(node.op for node in meta_graph.graph_def.node)
    # Go through the functions in the graph definition
    for func in meta_graph.graph_def.library.function:
        # Add operations in each function
        model_op_names.update(node.op for node in func.node_def)
# Convert to list, sorted if you want
model_op_names = sorted(model_op_names)
print(*model_op_names, sep="\n")
